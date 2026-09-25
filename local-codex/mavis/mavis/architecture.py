"""Static ownership checks for Mavis's model-gateway boundary.

This is deliberately a narrow source guard, not a general Python security proof.
Run it when Mavis source or dependencies change.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib


# Mavis owns orchestration, never provider SDKs or provider discovery.
FORBIDDEN_DISTRIBUTIONS = frozenset({
    "openai", "anthropic", "google-genai", "google-generativeai", "boto3",
    "botocore", "cohere", "mistralai", "groq", "together", "litellm",
    "openrouter", "model-gateway", "requests", "httpx", "aiohttp", "urllib3",
})
FORBIDDEN_MODULES = frozenset({
    "openai", "anthropic", "google.genai", "google.generativeai", "boto3",
    "botocore", "cohere", "mistralai", "groq", "together", "litellm",
    "openrouter", "model_gateway",
})
NETWORK_MODULES = frozenset({"requests", "httpx", "aiohttp", "http.client", "urllib.request"})
# These functions call only the local oMLX HTTP endpoint. Any new caller or
# transport must receive a deliberate review here and in the function itself.
LOCAL_HTTP = {
    "runtime.py": {"request_json", "iris_voice_session_active"},
    "evaluations.py": {"_post_json"},
    "helper_eval.py": {"_completion"},
}
LOCAL_URL_BUILDERS = {
    "runtime.py": "_origin",
    "evaluations.py": "_local_response_url",
    "helper_eval.py": "_local_endpoint",
}
PROVIDER_NAMES = frozenset({
    "openai", "anthropic", "google", "gemini", "minimax", "zai", "glm",
    "mistral", "cohere", "groq", "together", "openrouter", "kimi",
})


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


def _matches(module: str, names: frozenset[str]) -> bool:
    return any(module == name or module.startswith(name + ".") for name in names)


def _distribution(requirement: str) -> str:
    # PEP 508's distribution name ends before extras, a version specifier, or
    # environment marker. Reject an unparseable entry rather than skipping it.
    name = requirement.split(";", 1)[0].strip().split("[", 1)[0]
    for index, character in enumerate(name):
        if character in "<>=!~ @":
            name = name[:index]
            break
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(f"invalid dependency name: {requirement!r}")
    return name.lower().replace("_", "-").replace(".", "-")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _resolve_call(call: str | None, aliases: dict[str, str]) -> str | None:
    if not call:
        return None
    head, dot, tail = call.partition(".")
    return aliases.get(head, head) + (dot + tail if dot else "")


def _literal_provider_map(node: ast.AST) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    keys = {key.value.lower() for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    return len(keys & PROVIDER_NAMES) >= 2


def _registry_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in {"providers", "model_providers", "provider_routes", "provider_configs"}:
        return True
    return "provider" in lowered and any(part in lowered for part in
                                          ("registry", "catalog", "clients", "directory", "map"))


def _owner(tree: ast.AST, node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next((parent for parent in ast.walk(tree)
                 if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node in ast.walk(parent)), None)


def _safe_request(node: ast.Call, filename: str) -> bool:
    if not node.args:
        return False
    url = node.args[0]
    if isinstance(url, ast.BinOp) and isinstance(url.op, ast.Add):
        url = url.left
    return isinstance(url, ast.Call) and _call_name(url.func) == LOCAL_URL_BUILDERS.get(filename)


def _request_binding(owner: ast.AST, name: str, filename: str,
                     aliases: dict[str, str]) -> bool:
    return any(
        isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in item.targets)
        and isinstance(item.value, ast.Call)
        and _resolve_call(_call_name(item.value.func), aliases) == "urllib.request.Request"
        and _safe_request(item.value, filename)
        for item in ast.walk(owner)
    )


def _network_command(node: ast.Call, resolved: str | None) -> bool:
    if resolved not in {"subprocess.run", "subprocess.Popen", "subprocess.call",
                        "subprocess.check_call", "subprocess.check_output", "os.system"}:
        return False
    if not node.args:
        return False
    command = node.args[0]
    if isinstance(command, (ast.List, ast.Tuple)) and command.elts:
        command = command.elts[0]
    if not isinstance(command, ast.Constant) or not isinstance(command.value, str):
        return False
    return command.value.split(maxsplit=1)[0].rsplit("/", 1)[-1] in {"curl", "wget", "http", "https"}


def check_source(path: Path, *, display: str | None = None) -> list[Finding]:
    label = display or path.name
    # A nested package file named runtime.py does not inherit the HTTP exception.
    local_file = label.removeprefix("mavis/")
    if "/" in local_file:
        local_file = ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=label)
    except (OSError, UnicodeError, SyntaxError) as error:
        return [Finding(label, getattr(error, "lineno", 1) or 1, "syntax", str(error))]
    findings: list[Finding] = []
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
                if _matches(alias.name, FORBIDDEN_MODULES):
                    findings.append(Finding(label, node.lineno, "forbidden-import", alias.name))
                if _matches(alias.name, NETWORK_MODULES) and not (
                    alias.name == "urllib.request" and local_file in LOCAL_HTTP
                ):
                    findings.append(Finding(label, node.lineno, "network-import", alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                aliases[alias.asname or alias.name] = full
                if _matches(node.module, FORBIDDEN_MODULES) or _matches(full, FORBIDDEN_MODULES):
                    findings.append(Finding(label, node.lineno, "forbidden-import", full))
                if (_matches(node.module, NETWORK_MODULES) or
                        _matches(full, NETWORK_MODULES)) and not (
                    node.module == "urllib.request" and local_file in LOCAL_HTTP
                    or node.module == "urllib" and alias.name == "request" and local_file in LOCAL_HTTP
                ):
                    findings.append(Finding(label, node.lineno, "network-import", full))
        elif isinstance(node, ast.Call):
            call = _call_name(node.func)
            resolved = _resolve_call(call, aliases)
            if resolved in {"__import__", "importlib.import_module"} and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if _matches(value.value, FORBIDDEN_MODULES | NETWORK_MODULES):
                        findings.append(Finding(label, node.lineno, "dynamic-import", value.value))
            if _network_command(node, resolved):
                findings.append(Finding(label, node.lineno, "network-process", "HTTP command bypasses model gateway"))
            if resolved and resolved.endswith(".urlopen"):
                owner = _owner(tree, node)
                if (not owner or owner.name not in LOCAL_HTTP.get(local_file, set())
                        or not node.args or not isinstance(node.args[0], ast.Name)
                        or not _request_binding(owner, node.args[0].id, local_file, aliases)):
                    findings.append(Finding(label, node.lineno, "network-call", "urlopen outside bounded local request"))
            if resolved == "urllib.request.Request" and local_file in LOCAL_HTTP:
                owner = _owner(tree, node)
                if not owner or owner.name not in LOCAL_HTTP[local_file] or not _safe_request(node, local_file):
                    findings.append(Finding(label, node.lineno, "unbounded-url", "Request lacks local URL builder"))
        elif _literal_provider_map(node):
            findings.append(Finding(label, node.lineno, "provider-registry", "multi-provider mapping belongs to model gateway"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and _registry_name(target.id) for target in targets):
                findings.append(Finding(label, node.lineno, "provider-registry", "provider catalog belongs to model gateway"))
    return findings


def check_project(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    manifest = root / "pyproject.toml"
    try:
        manifest_data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        project = manifest_data["project"]
        if not isinstance(project, dict):
            raise ValueError("project table is invalid")
        direct = project.get("dependencies", [])
        build = manifest_data.get("build-system", {})
        optional = project.get("optional-dependencies", {})
        if not isinstance(build, dict) or not isinstance(optional, dict):
            raise ValueError("dependency table is invalid")
        build_requires = build.get("requires", [])
        if not isinstance(direct, list) or not isinstance(build_requires, list):
            raise ValueError("dependency list is invalid")
        requirements = list(direct) + list(build_requires)
        for group in optional.values():
            if not isinstance(group, list):
                raise ValueError("optional dependency group is invalid")
            requirements.extend(group)
        for requirement in requirements:
            if not isinstance(requirement, str):
                raise ValueError("dependency is not a string")
            name = _distribution(requirement)
            if name in FORBIDDEN_DISTRIBUTIONS:
                findings.append(Finding("pyproject.toml", 1, "forbidden-dependency", name))
    except (OSError, UnicodeError, ValueError, KeyError, tomllib.TOMLDecodeError) as error:
        findings.append(Finding("pyproject.toml", 1, "manifest", str(error)))
    source = root / "mavis"
    if not source.is_dir():
        findings.append(Finding("mavis", 1, "source", "package directory is missing"))
    else:
        for path in sorted(source.rglob("*.py")):
            findings.extend(check_source(path, display=path.relative_to(root).as_posix()))
    return sorted(findings, key=lambda item: (item.path, item.line, item.rule, item.detail))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = check_project(root)
    for finding in findings:
        print(finding)
    if findings:
        return 1
    print("Mavis architecture boundary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
