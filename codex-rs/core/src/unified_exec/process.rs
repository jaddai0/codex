#![allow(clippy::module_inception)]

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use tokio::sync::Mutex;
use tokio::sync::Notify;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::sync::oneshot::error::TryRecvError;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;

use codex_exec_server::ExecProcess;
use codex_exec_server::ExecProcessEvent;
use codex_exec_server::ProcessSignal as ExecServerProcessSignal;
use codex_exec_server::ReadResponse as ExecReadResponse;
use codex_exec_server::StartedExecProcess;
use codex_exec_server::WriteStatus;
use codex_protocol::exec_output::ExecToolCallOutput;
use codex_protocol::exec_output::StreamOutput;
use codex_protocol::protocol::TruncationPolicy;
use codex_sandboxing::SandboxType;
use codex_sandboxing::is_likely_sandbox_denied;
use codex_sandboxing::record_filesystem_sandbox_violation;
use codex_utils_output_truncation::formatted_truncate_text;
use codex_utils_pty::ExecCommandSession;
use codex_utils_pty::ProcessSignal as PtyProcessSignal;
use codex_utils_pty::SpawnedPty;

use super::UNIFIED_EXEC_OUTPUT_MAX_BYTES;
use super::UNIFIED_EXEC_OUTPUT_MAX_TOKENS;
use super::UnifiedExecError;
use super::head_tail_buffer::HeadTailBuffer;
use super::process_state::ProcessState;
use super::raw_output_spool::RawOutputReference;
use super::raw_output_spool::RawOutputSpool;
use crate::shell_snapshot::ShellSnapshotFile;

const EARLY_EXIT_GRACE_PERIOD: Duration = Duration::from_millis(150);

fn append_raw_output(
    spool: &Option<Arc<StdMutex<RawOutputSpool>>>,
    chunk: &[u8],
) -> std::io::Result<()> {
    let Some(spool) = spool else {
        return Ok(());
    };
    let mut spool = spool
        .lock()
        .map_err(|_| std::io::Error::other("raw output spool lock poisoned"))?;
    spool.append(chunk)
}

fn replay_has_unrecoverable_output_gap(
    last_seq: u64,
    chunk_sequences: impl IntoIterator<Item = u64>,
    next_seq: u64,
    unseen_exit: bool,
    unseen_close: bool,
) -> bool {
    let mut expected = last_seq.saturating_add(1);
    let mut missing = 0_u64;
    for seq in chunk_sequences {
        if seq >= expected {
            missing = missing.saturating_add(seq - expected);
            expected = seq.saturating_add(1);
        }
    }
    // The executor's next_seq includes terminal events, which never carry
    // output. An exit can precede a final output chunk, so count gaps across
    // the whole replay rather than requiring adjacent chunk sequences.
    missing = missing.saturating_add(next_seq.saturating_sub(expected));
    let terminal_events = u64::from(unseen_exit) + u64::from(unseen_close);
    missing > terminal_events
}

enum LocalOutputReceiver {
    Ordinary(broadcast::Receiver<Vec<u8>>),
    Mavis(mpsc::Receiver<Vec<u8>>),
}

fn combine_mavis_output_receivers(
    mut stdout_rx: mpsc::Receiver<Vec<u8>>,
    mut stderr_rx: mpsc::Receiver<Vec<u8>>,
) -> mpsc::Receiver<Vec<u8>> {
    let (combined_tx, combined_rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let mut stdout_open = true;
        let mut stderr_open = true;
        loop {
            tokio::select! {
                stdout = stdout_rx.recv(), if stdout_open => match stdout {
                    Some(chunk) => if combined_tx.send(chunk).await.is_err() { break; },
                    None => stdout_open = false,
                },
                stderr = stderr_rx.recv(), if stderr_open => match stderr {
                    Some(chunk) => if combined_tx.send(chunk).await.is_err() { break; },
                    None => stderr_open = false,
                },
                else => break,
            }
        }
    });
    combined_rx
}
pub(crate) trait SpawnLifecycle: std::fmt::Debug + Send + Sync {
    /// Returns file descriptors that must stay open across the child `exec()`.
    ///
    /// The returned descriptors must already be valid in the parent process and
    /// stay valid until `after_spawn()` runs, which is the first point where
    /// the parent may release its copies.
    fn inherited_fds(&self) -> Vec<i32> {
        Vec::new()
    }

    fn after_spawn(&mut self) {}
}

pub(crate) type SpawnLifecycleHandle = Box<dyn SpawnLifecycle>;

#[derive(Debug, Default)]
/// Spawn lifecycle that performs no extra setup around process launch.
pub(crate) struct NoopSpawnLifecycle;

impl SpawnLifecycle for NoopSpawnLifecycle {}

/// Shared output state exposed to polling and streaming consumers.
#[derive(Clone)]
pub(crate) struct OutputHandles<const MAX_BYTES: usize = UNIFIED_EXEC_OUTPUT_MAX_BYTES> {
    pub(crate) output_buffer: Arc<Mutex<HeadTailBuffer<MAX_BYTES>>>,
    pub(crate) output_notify: Arc<Notify>,
    pub(crate) output_closed: Arc<AtomicBool>,
    pub(crate) output_closed_notify: Arc<Notify>,
    pub(crate) cancellation_token: CancellationToken,
}

struct OutputTaskGuard {
    output_closed: Arc<AtomicBool>,
    output_closed_notify: Arc<Notify>,
}

impl Drop for OutputTaskGuard {
    fn drop(&mut self) {
        self.output_closed.store(true, Ordering::Release);
        self.output_closed_notify.notify_waiters();
    }
}

/// Transport-specific process handle used by unified exec.
enum ProcessHandle {
    Local(Box<ExecCommandSession>),
    ExecServer(Arc<dyn ExecProcess>),
}

/// Unified wrapper over directly spawned PTY sessions and exec-server-backed
/// processes.
pub(crate) struct UnifiedExecProcess {
    process_handle: ProcessHandle,
    output_tx: broadcast::Sender<Vec<u8>>,
    output: OutputHandles,
    raw_output_spool: Option<Arc<StdMutex<RawOutputSpool>>>,
    output_drained: Arc<Notify>,
    interaction_lock: Arc<Mutex<()>>,
    state_tx: watch::Sender<ProcessState>,
    state_rx: watch::Receiver<ProcessState>,
    output_task: Option<JoinHandle<()>>,
    sandbox_type: SandboxType,
    timed_out: AtomicBool,
    _spawn_lifecycle: Option<SpawnLifecycleHandle>,
    // The shell may still need to replay this file after process startup returns.
    pub(crate) _shell_snapshot: Option<Arc<ShellSnapshotFile>>,
}

impl std::fmt::Debug for UnifiedExecProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedExecProcess")
            .field("has_exited", &self.has_exited())
            .field("exit_code", &self.exit_code())
            .field("sandbox_type", &self.sandbox_type)
            .finish_non_exhaustive()
    }
}

impl UnifiedExecProcess {
    fn new(
        process_handle: ProcessHandle,
        sandbox_type: SandboxType,
        spawn_lifecycle: Option<SpawnLifecycleHandle>,
        raw_output_spool: Option<RawOutputSpool>,
    ) -> Self {
        let raw_output_spool = raw_output_spool.map(|spool| Arc::new(StdMutex::new(spool)));
        let output = OutputHandles {
            output_buffer: Arc::new(Mutex::new(HeadTailBuffer::default())),
            output_notify: Arc::new(Notify::new()),
            output_closed: Arc::new(AtomicBool::new(false)),
            output_closed_notify: Arc::new(Notify::new()),
            cancellation_token: CancellationToken::new(),
        };
        let output_drained = Arc::new(Notify::new());
        let (output_tx, _) = broadcast::channel(64);
        let (state_tx, state_rx) = watch::channel(ProcessState::default());

        Self {
            process_handle,
            output_tx,
            output,
            raw_output_spool,
            output_drained,
            interaction_lock: Arc::new(Mutex::new(())),
            state_tx,
            state_rx,
            output_task: None,
            sandbox_type,
            timed_out: AtomicBool::new(false),
            _spawn_lifecycle: spawn_lifecycle,
            _shell_snapshot: None,
        }
    }

    pub(super) fn raw_output_reference(
        &self,
    ) -> Result<Option<RawOutputReference>, UnifiedExecError> {
        let Some(spool) = &self.raw_output_spool else {
            return Ok(None);
        };
        let mut spool = spool.lock().map_err(|_| {
            UnifiedExecError::process_failed("raw output spool lock poisoned".to_string())
        })?;
        let complete =
            self.output.output_closed.load(Ordering::Acquire) && self.failure_message().is_none();
        spool.reference(complete).map(Some).map_err(|err| {
            UnifiedExecError::process_failed(format!("raw output spool sync: {err}"))
        })
    }

    pub(super) async fn write(&self, data: &[u8]) -> Result<(), UnifiedExecError> {
        match &self.process_handle {
            ProcessHandle::Local(process_handle) => process_handle
                .writer_sender()
                .send(data.to_vec())
                .await
                .map_err(|_| UnifiedExecError::WriteToStdin),
            ProcessHandle::ExecServer(process_handle) => {
                match process_handle.write(data.to_vec()).await {
                    Ok(response) => match response.status {
                        WriteStatus::Accepted => Ok(()),
                        WriteStatus::UnknownProcess | WriteStatus::StdinClosed => {
                            let state = self.state_rx.borrow().clone();
                            let _ = self.state_tx.send_replace(state.exited(state.exit_code));
                            self.output.cancellation_token.cancel();
                            Err(UnifiedExecError::WriteToStdin)
                        }
                        WriteStatus::Starting => Err(UnifiedExecError::WriteToStdin),
                    },
                    Err(err) => Err(UnifiedExecError::process_failed(err.to_string())),
                }
            }
        }
    }

    pub(super) fn output_handles(&self) -> &OutputHandles {
        &self.output
    }

    pub(super) fn output_receiver(&self) -> tokio::sync::broadcast::Receiver<Vec<u8>> {
        self.output_tx.subscribe()
    }

    pub(super) fn cancellation_token(&self) -> CancellationToken {
        self.output.cancellation_token.clone()
    }

    pub(super) fn output_drained_notify(&self) -> Arc<Notify> {
        Arc::clone(&self.output_drained)
    }

    pub(super) fn interaction_lock(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.interaction_lock)
    }

    pub(super) fn has_exited(&self) -> bool {
        let state = self.state_rx.borrow().clone();
        match &self.process_handle {
            ProcessHandle::Local(process_handle) => state.has_exited || process_handle.has_exited(),
            ProcessHandle::ExecServer(_) => state.has_exited,
        }
    }

    pub(super) fn exit_code(&self) -> Option<i32> {
        if self.timed_out() {
            return Some(124);
        }
        let state = self.state_rx.borrow().clone();
        match &self.process_handle {
            ProcessHandle::Local(process_handle) => {
                state.exit_code.or_else(|| process_handle.exit_code())
            }
            ProcessHandle::ExecServer(_) => state.exit_code,
        }
    }

    pub(super) fn mark_timed_out(&self) {
        self.timed_out.store(true, Ordering::Release);
    }

    pub(super) fn timed_out(&self) -> bool {
        self.timed_out.load(Ordering::Acquire)
    }

    fn finish_termination(&self) {
        self.output.cancellation_token.cancel();
        if let Some(output_task) = &self.output_task {
            output_task.abort();
        }
    }

    pub(super) fn terminate(&self) {
        match &self.process_handle {
            ProcessHandle::Local(process_handle) => process_handle.terminate(),
            ProcessHandle::ExecServer(process_handle) => {
                let process_handle = Arc::clone(process_handle);
                tokio::spawn(async move {
                    let _ = process_handle.terminate().await;
                });
            }
        }
        self.finish_termination();
    }

    pub(super) async fn terminate_confirmed(&self) -> Result<(), UnifiedExecError> {
        match &self.process_handle {
            ProcessHandle::Local(process_handle) => process_handle.terminate(),
            ProcessHandle::ExecServer(process_handle) => {
                process_handle
                    .terminate()
                    .await
                    .map_err(|err| UnifiedExecError::process_failed(err.to_string()))?;
            }
        }
        self.signal_exit(self.exit_code());
        self.finish_termination();
        Ok(())
    }

    pub(super) async fn interrupt(&self) -> Result<(), UnifiedExecError> {
        match &self.process_handle {
            ProcessHandle::Local(process_handle) => process_handle
                .signal(PtyProcessSignal::Interrupt)
                .map_err(|err| UnifiedExecError::process_failed(err.to_string())),
            ProcessHandle::ExecServer(process_handle) => process_handle
                .signal(ExecServerProcessSignal::Interrupt)
                .await
                .map_err(|err| UnifiedExecError::process_failed(err.to_string())),
        }
    }

    pub(super) fn fail_and_terminate(&self, message: String) {
        let state = self.state_rx.borrow().clone();
        if state.failure_message.is_none() {
            let _ = self.state_tx.send_replace(state.failed(message));
        }
        self.terminate();
    }

    async fn snapshot_output(&self) -> Vec<u8> {
        let guard = self.output.output_buffer.lock().await;
        guard.to_bytes()
    }

    pub(crate) fn sandbox_type(&self) -> SandboxType {
        self.sandbox_type
    }

    pub(super) fn failure_message(&self) -> Option<String> {
        self.state_rx.borrow().failure_message.clone()
    }

    pub(super) async fn check_for_sandbox_denial(&self) -> Result<(), UnifiedExecError> {
        let _ = tokio::time::timeout(
            Duration::from_millis(20),
            self.output.output_notify.notified(),
        )
        .await;

        let aggregated = self.snapshot_output().await;
        let aggregated_text = String::from_utf8_lossy(&aggregated);
        self.check_for_sandbox_denial_with_text(aggregated_text.as_ref())
            .await?;

        Ok(())
    }

    pub(super) async fn check_for_sandbox_denial_with_text(
        &self,
        text: &str,
    ) -> Result<(), UnifiedExecError> {
        let executor_reported_denial = self.state_rx.borrow().sandbox_denied;
        let sandbox_type = self.sandbox_type();
        if !self.has_exited() || (!executor_reported_denial && sandbox_type == SandboxType::None) {
            return Ok(());
        }

        let exit_code = self.exit_code().unwrap_or(-1);
        let exec_output = ExecToolCallOutput {
            exit_code,
            stderr: StreamOutput::new(text.to_string()),
            aggregated_output: StreamOutput::new(text.to_string()),
            ..Default::default()
        };
        let likely_sandbox_denial = is_likely_sandbox_denied(sandbox_type, &exec_output);
        if likely_sandbox_denial {
            record_filesystem_sandbox_violation(sandbox_type, &exec_output);
        }
        if executor_reported_denial || likely_sandbox_denial {
            let snippet = formatted_truncate_text(
                text,
                TruncationPolicy::Tokens(UNIFIED_EXEC_OUTPUT_MAX_TOKENS),
            );
            let message = if snippet.is_empty() {
                format!("Process exited with code {exit_code}")
            } else {
                snippet
            };
            return Err(UnifiedExecError::sandbox_denied(message, exec_output));
        }
        Ok(())
    }

    pub(super) async fn from_spawned(
        spawned: SpawnedPty,
        sandbox_type: SandboxType,
        spawn_lifecycle: SpawnLifecycleHandle,
        raw_output_spool: Option<RawOutputSpool>,
    ) -> Result<Self, UnifiedExecError> {
        let SpawnedPty {
            session: process_handle,
            stdout_rx,
            stderr_rx,
            mut exit_rx,
        } = spawned;
        let mut managed = Self::new(
            ProcessHandle::Local(Box::new(process_handle)),
            sandbox_type,
            Some(spawn_lifecycle),
            raw_output_spool,
        );
        let output_rx = if managed.raw_output_spool.is_some() {
            LocalOutputReceiver::Mavis(combine_mavis_output_receivers(stdout_rx, stderr_rx))
        } else {
            LocalOutputReceiver::Ordinary(codex_utils_pty::combine_output_receivers(
                stdout_rx, stderr_rx,
            ))
        };
        managed.output_task = Some(Self::spawn_local_output_task(
            output_rx,
            managed.output_handles().clone(),
            managed.output_tx.clone(),
            managed.state_tx.clone(),
            managed.raw_output_spool.clone(),
        ));

        match exit_rx.try_recv() {
            Ok(exit_code) => {
                managed.signal_exit(Some(exit_code));
                managed.check_for_sandbox_denial().await?;
                return Ok(managed);
            }
            Err(TryRecvError::Closed) => {
                managed.signal_exit(/*exit_code*/ None);
                managed.check_for_sandbox_denial().await?;
                return Ok(managed);
            }
            Err(TryRecvError::Empty) => {}
        }

        if let Ok(exit_result) = tokio::time::timeout(EARLY_EXIT_GRACE_PERIOD, &mut exit_rx).await {
            managed.signal_exit(exit_result.ok());
            managed.check_for_sandbox_denial().await?;
            return Ok(managed);
        }

        tokio::spawn({
            let state_tx = managed.state_tx.clone();
            let cancellation_token = managed.output.cancellation_token.clone();
            async move {
                let exit_code = exit_rx.await.ok();
                let state = state_tx.borrow().clone();
                let _ = state_tx.send_replace(state.exited(exit_code));
                cancellation_token.cancel();
            }
        });

        Ok(managed)
    }

    pub(super) async fn from_exec_server_started(
        started: StartedExecProcess,
        raw_output_spool: Option<RawOutputSpool>,
    ) -> Result<Self, UnifiedExecError> {
        let mavis_required = std::env::var("MAVIS_RAW_OUTPUT_REQUIRED").as_deref() == Ok("1");
        ensure_mavis_exec_server_output_spool(&started, mavis_required).await?;
        let process_handle = ProcessHandle::ExecServer(Arc::clone(&started.process));
        // Older peers do not report this field. In that case, skip local
        // classification rather than attributing a violation to a guessed backend.
        let sandbox_type = started.sandbox_type.unwrap_or(SandboxType::None);
        let mut managed = Self::new(
            process_handle,
            sandbox_type,
            /*spawn_lifecycle*/ None,
            raw_output_spool,
        );
        let output_handles = managed.output_handles().clone();
        managed.output_task = Some(Self::spawn_exec_server_output_task(
            started,
            output_handles,
            managed.output_tx.clone(),
            managed.state_tx.clone(),
            managed.raw_output_spool.clone(),
        ));

        let mut state_rx = managed.state_rx.clone();
        if tokio::time::timeout(EARLY_EXIT_GRACE_PERIOD, async {
            loop {
                let state = state_rx.borrow().clone();
                if state.has_exited || state.failure_message.is_some() {
                    break;
                }
                if state_rx.changed().await.is_err() {
                    break;
                }
            }
        })
        .await
        .is_ok()
        {
            managed.check_for_sandbox_denial().await?;
        }

        Ok(managed)
    }

    fn spawn_exec_server_output_task(
        started: StartedExecProcess,
        output_handles: OutputHandles,
        output_tx: broadcast::Sender<Vec<u8>>,
        state_tx: watch::Sender<ProcessState>,
        raw_output_spool: Option<Arc<StdMutex<RawOutputSpool>>>,
    ) -> JoinHandle<()> {
        let OutputHandles {
            output_buffer,
            output_notify,
            output_closed,
            output_closed_notify,
            cancellation_token,
        } = output_handles;
        let process = started.process;
        let mut events = process.subscribe_events();
        tokio::spawn(async move {
            let _output_task_guard = OutputTaskGuard {
                output_closed: Arc::clone(&output_closed),
                output_closed_notify: Arc::clone(&output_closed_notify),
            };
            let mut last_seq: u64 = 0;
            'output_loop: loop {
                let event = match events.recv().await {
                    Ok(event) => Some(event),
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        if raw_output_spool.is_some() {
                            let state = state_tx.borrow().clone();
                            let _ = state_tx.send_replace(
                                state.failed("raw output event stream lagged".to_string()),
                            );
                            cancellation_token.cancel();
                            break;
                        }
                        None
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        let state = state_tx.borrow().clone();
                        let _ = state_tx.send_replace(
                            state.failed("exec-server process event stream closed".to_string()),
                        );
                        output_closed.store(true, Ordering::Release);
                        output_closed_notify.notify_waiters();
                        cancellation_token.cancel();
                        break;
                    }
                };
                let event_seq = event.as_ref().and_then(|event| match event {
                    ExecProcessEvent::Output(chunk) => Some(chunk.seq),
                    ExecProcessEvent::Exited { seq, .. } | ExecProcessEvent::Closed { seq } => {
                        Some(*seq)
                    }
                    ExecProcessEvent::Failed(_) => None,
                });
                let missing_sandbox_denial = matches!(
                    event.as_ref(),
                    Some(ExecProcessEvent::Exited {
                        sandbox_denied: None,
                        ..
                    })
                );
                if event.is_none()
                    || event_seq.is_some_and(|seq| seq > last_seq.saturating_add(1))
                    || missing_sandbox_denial
                {
                    let response = match process
                        .read(
                            Some(last_seq),
                            /*max_bytes*/ None,
                            /*wait_ms*/ Some(0),
                        )
                        .await
                    {
                        Ok(response) => response,
                        Err(err) => {
                            let state = state_tx.borrow().clone();
                            let _ = state_tx.send_replace(state.failed(err.to_string()));
                            output_closed.store(true, Ordering::Release);
                            output_closed_notify.notify_waiters();
                            cancellation_token.cancel();
                            break;
                        }
                    };
                    let ExecReadResponse {
                        chunks,
                        next_seq,
                        exited,
                        exit_code,
                        closed,
                        failure,
                        sandbox_denied,
                    } = response;
                    if raw_output_spool.is_some()
                        && replay_has_unrecoverable_output_gap(
                            last_seq,
                            chunks.iter().map(|chunk| chunk.seq),
                            next_seq,
                            exited && !state_tx.borrow().has_exited,
                            closed && !output_closed.load(Ordering::Acquire),
                        )
                    {
                        let state = state_tx.borrow().clone();
                        let _ = state_tx.send_replace(state.failed(
                            "raw output stream has an unrecoverable sequence gap".to_string(),
                        ));
                        cancellation_token.cancel();
                        break;
                    }
                    let prior_seq = last_seq;
                    for chunk in chunks.into_iter().filter(|chunk| chunk.seq > prior_seq) {
                        last_seq = chunk.seq;
                        let bytes = chunk.chunk.into_inner();
                        if let Err(err) = append_raw_output(&raw_output_spool, &bytes) {
                            let state = state_tx.borrow().clone();
                            let _ = state_tx.send_replace(state.failed(err.to_string()));
                            cancellation_token.cancel();
                            break 'output_loop;
                        }
                        let mut guard = output_buffer.lock().await;
                        guard.push_chunk(&bytes);
                        drop(guard);
                        let _ = output_tx.send(bytes);
                        output_notify.notify_waiters();
                    }
                    last_seq = last_seq.max(next_seq.saturating_sub(1));
                    if let Some(message) = failure {
                        let state = state_tx.borrow().clone();
                        let _ = state_tx.send_replace(state.failed(message));
                        output_closed.store(true, Ordering::Release);
                        output_closed_notify.notify_waiters();
                        cancellation_token.cancel();
                        break;
                    }
                    if sandbox_denied || exited {
                        let mut state = state_tx.borrow().clone();
                        state.sandbox_denied |= sandbox_denied;
                        let _ = state_tx.send_replace(if exited {
                            state.exited(exit_code)
                        } else {
                            state
                        });
                    }
                    if closed {
                        output_closed.store(true, Ordering::Release);
                        output_closed_notify.notify_waiters();
                        cancellation_token.cancel();
                        break;
                    }
                    continue;
                }

                let Some(event) = event else {
                    continue;
                };
                match event {
                    ExecProcessEvent::Output(chunk) => {
                        if chunk.seq <= last_seq {
                            continue;
                        }
                        last_seq = chunk.seq;
                        let bytes = chunk.chunk.into_inner();
                        if let Err(err) = append_raw_output(&raw_output_spool, &bytes) {
                            let state = state_tx.borrow().clone();
                            let _ = state_tx.send_replace(state.failed(err.to_string()));
                            cancellation_token.cancel();
                            break;
                        }
                        let mut guard = output_buffer.lock().await;
                        guard.push_chunk(&bytes);
                        drop(guard);
                        let _ = output_tx.send(bytes);
                        output_notify.notify_waiters();
                    }
                    ExecProcessEvent::Exited {
                        seq,
                        exit_code,
                        sandbox_denied,
                    } => {
                        if seq <= last_seq {
                            continue;
                        }
                        last_seq = seq;
                        let mut state = state_tx.borrow().clone();
                        state.sandbox_denied |= sandbox_denied.unwrap_or(false);
                        let _ = state_tx.send_replace(state.exited(Some(exit_code)));
                    }
                    ExecProcessEvent::Closed { seq } => {
                        if seq <= last_seq {
                            continue;
                        }
                        output_closed.store(true, Ordering::Release);
                        output_closed_notify.notify_waiters();
                        cancellation_token.cancel();
                        break;
                    }
                    ExecProcessEvent::Failed(message) => {
                        let state = state_tx.borrow().clone();
                        let _ = state_tx.send_replace(state.failed(message));
                        output_closed.store(true, Ordering::Release);
                        output_closed_notify.notify_waiters();
                        cancellation_token.cancel();
                        break;
                    }
                }
            }
        })
    }

    fn spawn_local_output_task(
        mut receiver: LocalOutputReceiver,
        output_handles: OutputHandles,
        output_tx: broadcast::Sender<Vec<u8>>,
        state_tx: watch::Sender<ProcessState>,
        raw_output_spool: Option<Arc<StdMutex<RawOutputSpool>>>,
    ) -> JoinHandle<()> {
        let OutputHandles {
            output_buffer,
            output_notify,
            output_closed,
            output_closed_notify,
            ..
        } = output_handles;
        tokio::spawn(async move {
            let _output_task_guard = OutputTaskGuard {
                output_closed: Arc::clone(&output_closed),
                output_closed_notify: Arc::clone(&output_closed_notify),
            };
            loop {
                let next = match &mut receiver {
                    LocalOutputReceiver::Ordinary(receiver) => receiver.recv().await.map(Some),
                    LocalOutputReceiver::Mavis(receiver) => Ok(receiver.recv().await),
                };
                match next {
                    Ok(Some(chunk)) => {
                        if let Err(err) = append_raw_output(&raw_output_spool, &chunk) {
                            let state = state_tx.borrow().clone();
                            let _ = state_tx.send_replace(state.failed(err.to_string()));
                            break;
                        }
                        let mut guard = output_buffer.lock().await;
                        guard.push_chunk(&chunk);
                        drop(guard);
                        let _ = output_tx.send(chunk);
                        output_notify.notify_waiters();
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        if raw_output_spool.is_some() {
                            let state = state_tx.borrow().clone();
                            let _ = state_tx
                                .send_replace(state.failed("raw output stream lagged".to_string()));
                            break;
                        }
                    }
                    Ok(None) | Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        output_closed.store(true, Ordering::Release);
                        output_closed_notify.notify_waiters();
                        break;
                    }
                };
            }
        })
    }

    fn signal_exit(&self, exit_code: Option<i32>) {
        let state = self.state_rx.borrow().clone();
        let _ = self.state_tx.send_replace(state.exited(exit_code));
        self.output.cancellation_token.cancel();
    }
}

pub(super) async fn ensure_mavis_exec_server_output_spool(
    started: &StartedExecProcess,
    required: bool,
) -> Result<(), UnifiedExecError> {
    if required && !started.mavis_raw_output_active {
        let _ = started.process.terminate().await;
        return Err(UnifiedExecError::process_failed(
            "exec-server did not confirm Mavis raw output spool".to_string(),
        ));
    }
    Ok(())
}

impl Drop for UnifiedExecProcess {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(test)]
mod raw_output_tests {
    use super::*;

    #[test]
    fn terminal_only_replay_cannot_hide_evicted_output() {
        // Output seq 1 was evicted; only Exited seq 2 and Closed seq 3 remain.
        assert!(replay_has_unrecoverable_output_gap(0, [], 4, true, true));
        // A contiguous output followed by two terminal events is complete.
        assert!(!replay_has_unrecoverable_output_gap(0, [1], 4, true, true));
        // Exit seq 1 may arrive before the final output chunk seq 2.
        assert!(!replay_has_unrecoverable_output_gap(0, [2], 3, true, false));
        // If Exited was already observed, it cannot excuse a later missing chunk.
        assert!(replay_has_unrecoverable_output_gap(2, [], 5, false, true));
    }

    #[tokio::test]
    async fn local_output_task_spools_full_stream_before_cap() {
        let temp = tempfile::tempdir().unwrap();
        let spool = Arc::new(StdMutex::new(
            RawOutputSpool::open_in(temp.path().join("tool-output")).unwrap(),
        ));
        let output = OutputHandles::<UNIFIED_EXEC_OUTPUT_MAX_BYTES> {
            output_buffer: Arc::new(Mutex::new(HeadTailBuffer::default())),
            output_notify: Arc::new(Notify::new()),
            output_closed: Arc::new(AtomicBool::new(false)),
            output_closed_notify: Arc::new(Notify::new()),
            cancellation_token: CancellationToken::new(),
        };
        let (source_tx, source_rx) = mpsc::channel(64);
        let (forward_tx, _) = broadcast::channel(64);
        let (state_tx, state_rx) = watch::channel(ProcessState::default());
        let task = UnifiedExecProcess::spawn_local_output_task(
            LocalOutputReceiver::Mavis(source_rx),
            output.clone(),
            forward_tx,
            state_tx,
            Some(Arc::clone(&spool)),
        );
        source_tx.send(vec![b'a'; 700_000]).await.unwrap();
        source_tx
            .send(b"\nFAILURE: buried diagnostic\n".to_vec())
            .await
            .unwrap();
        source_tx.send(vec![b'z'; 700_000]).await.unwrap();
        drop(source_tx);
        task.await.unwrap();

        assert!(state_rx.borrow().failure_message.is_none());
        assert!(output.output_closed.load(Ordering::Acquire));
        let reference = spool.lock().unwrap().reference(true).unwrap();
        let raw = std::fs::read(reference.path).unwrap();
        let needle = b"FAILURE: buried diagnostic";
        assert!(raw.windows(needle.len()).any(|window| window == needle));
        assert!(
            !output
                .output_buffer
                .lock()
                .await
                .to_bytes()
                .windows(needle.len())
                .any(|window| window == needle)
        );
    }
}
