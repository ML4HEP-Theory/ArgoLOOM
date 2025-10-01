#!/usr/bin/env python3
"""
agent-chat-bot.py
-----------------
Conversational front-end that can launch EITHER:
  • MG5 pipeline (API_MG5-kin.py)
  • CLASS pipeline (toolkit_class.py)
  • Physics KB search (toolkit_kb.py)

Key details
-----------
- Avoids empty `tool_calls` on assistant messages.
- Preserves proper tool->assistant threading (each tool execution yields a
  `{"role":"tool","tool_call_id":...}` message before the next model call).
- Includes a knowledge-base retrieval tool that queries a local FAISS index.

Usage
-----
export OPENAI_API_KEY=...
python agent-chat-bot.py \
  --mg5-path /path/to/mg5_aMC \
  --mg5-pipeline-file API_MG5-kin.py \
  --class-pipeline-file toolkit_class.py \
  --kb-index kb_out \
  --model gpt-4o
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
import toolkit_kb  # your KB toolkit module (kb_build/kb_query compatible)

# --------------------------- Startup Banner ---------------------------

def print_banner():
    banner = r"""
+-------------------------------------------------+
| || || || || || || || || || || || || || || || || |
| || || || || || || || || || || || || || || || || |
     _                    _     ___   ___  __  __ 
    / \   _ __ __ _  ___ | |   / _ \ / _ \|  \/  |
   / _ \ | '__/ _` |/ _ \| |  | | | | | | | |\/| |
  / ___ \| | | (_| | (_) | |__| |_| | |_| | |  | |
 /_/   \_\_|  \__, |\___/|_____\___/ \___/|_|  |_|
              |___/                               

| || || || || || || || || || || || || || || || || |
| || || || || || || || || || || || || || || || || |
+-------------------------------------------------+
|  A r g o L O O M  :  weaving Quarks → Cosmos    |
|  Linked Oracles  for  Observables and Models    |
+-------------------------------------------------+
    """
    print(banner)

# --------------------------- Defaults ---------------------------

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MG5_PIPELINE = "API_MG5-kin.py"
DEFAULT_CLASS_PIPELINE = "toolkit_class.py"

# --------------------------- Tool schemas ---------------------------

def build_tools_schema() -> List[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_mg5_pipeline",
                "description": "Run the MG5→LHE→analysis pipeline with the provided configuration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mg5_path": {"type": "string", "description": "Path to mg5_aMC executable"},
                        "import_model": {"type": "string", "default": "sm"},
                        "process": {"type": "string"},
                        "outtag": {"type": "string"},
                        "nevents": {"type": "integer", "minimum": 1, "default": 2000},
                        "ebeam_gev": {"type": "number", "exclusiveMinimum": 0},
                        "iseed": {"type": "integer", "default": 12345},
                        "pdg": {"type": "string", "description": "Comma PDG list (e.g. '11,-11')"},
                        "mode": {"type": "string", "enum": ["particle", "system"], "default": "system"},
                        "status": {"type": "integer", "default": 1},
                        "one_per_event": {"type": "boolean", "default": False},
                        "momentum_order": {"type": "string", "enum": ["Epxpypz", "pxpypzE"], "default": "Epxpypz"},
                        "rel_sigma_x": {"type": "number", "default": 0.02},
                        "rel_sigma_q2": {"type": "number", "default": 0.05},
                        "x_smear_alpha": {"type": "number", "default": 2.0},
                        "x_smear_beta": {"type": "number", "default": 2.0},
                        "plot": {"type": "boolean", "default": True},
                        "model_name": {"type": "string", "default": DEFAULT_MODEL}
                    },
                    "required": ["mg5_path", "import_model", "process", "outtag",
                                 "nevents", "ebeam_gev", "iseed", "pdg", "mode"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_class_pipeline",
                "description": "Run CLASS C_ell computation (ΛCDM + sterile neutrino options) and plot spectra.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "class_path": {"type": "string", "description": "Path to CLASS executable or class_public dir"},
                        "outtag": {"type": "string"},
                        "outdir": {"type": "string", "default": "."},
                        "workdir": {"type": "string", "default": "."},
                        "h": {"type": "number"},
                        "ombh2": {"type": "number"},
                        "omch2": {"type": "number"},
                        "As": {"type": "number"},
                        "ns": {"type": "number"},
                        "tau_reio": {"type": "number"},
                        "N_ur": {"type": "number", "nullable": True, "default": None},
                        "N_ncdm": {"type": "integer", "default": 0},
                        "m_ncdm": {"type": "string", "nullable": True, "default": None},
                        "T_ncdm": {"type": "string", "nullable": True, "default": None},
                        "deg_ncdm": {"type": "string", "nullable": True, "default": None},
                        "lensing": {"type": "string", "enum": ["yes", "no"], "default": "yes"},
                        "lmax": {"type": "integer", "default": 3000},
                        "output": {"type": "string", "default": "tCl,pCl,lCl"},
                        "plot": {"type": "boolean", "default": True}
                    },
                    "required": ["class_path", "outtag", "h", "ombh2", "omch2", "As", "ns", "tau_reio"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "kb_search",
                "description": "Search the local physics KB and return top-k supporting chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_dir": {"type": "string", "description": "Path to kb_out (manifest.json, chunks.jsonl, embeddings.npy, faiss.index)."},
                        "query": {"type": "string", "description": "User query / question."},
                        "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                        "engine": {"type": "string", "enum": ["auto", "faiss", "numpy"], "default": "auto"},
                        "doc_id": {"type": "string", "nullable": True, "description": "Optional doc id filter."}
                    },
                    "required": ["index_dir", "query"]
                }
            }
        }
    ]


# -------------------------- Arg builders ----------------------------

def cfg_to_argv_mg5(cfg: Dict[str, Any], pipeline_file: str) -> List[str]:
    args = [
        sys.executable, pipeline_file,
        "--mg5-path", cfg["mg5_path"],
        "--model-name", cfg.get("model_name", DEFAULT_MODEL),
        "--import-model", cfg["import_model"],
        "--process", cfg["process"],
        "--outtag", cfg["outtag"],
        "--nevents", str(cfg["nevents"]),
        "--ebeam-gev", str(cfg["ebeam_gev"]),
        "--iseed", str(cfg["iseed"]),
        "--pdg", cfg["pdg"],
        "--mode", cfg["mode"],
        "--status", str(cfg.get("status", 1)),
        "--momentum-order", cfg.get("momentum_order", "Epxpypz"),
        "--rel-sigma-x", str(cfg.get("rel_sigma_x", 0.02)),
        "--rel-sigma-q2", str(cfg.get("rel_sigma_q2", 0.05)),
        "--x-smear-alpha", str(cfg.get("x_smear_alpha", 2.0)),
        "--x-smear-beta", str(cfg.get("x_smear_beta", 2.0)),
    ]
    if cfg.get("one_per_event", False):
        args.append("--one-per-event")
    if cfg.get("plot", True):
        args.append("--plot")
    return args


def cfg_to_argv_class(cfg: Dict[str, Any], pipeline_file: str) -> List[str]:
    args = [
        sys.executable, pipeline_file,
        "--class-path", cfg["class_path"],
        "--outtag", cfg["outtag"],
        "--workdir", cfg.get("workdir", "."),
        "--outdir", cfg.get("outdir", "."),
        "--h", str(cfg["h"]),
        "--ombh2", str(cfg["ombh2"]),
        "--omch2", str(cfg["omch2"]),
        "--As", str(cfg["As"]),
        "--ns", str(cfg["ns"]),
        "--tau-reio", str(cfg["tau_reio"]),
        "--lensing", cfg.get("lensing", "yes"),
        "--lmax", str(cfg.get("lmax", 3000)),
        "--output", cfg.get("output", "tCl,pCl,lCl"),
    ]
    if cfg.get("N_ur") is not None:
        args += ["--N-ur", str(cfg["N_ur"])]
    if cfg.get("N_ncdm", 0) > 0:
        args += ["--N-ncdm", str(cfg["N_ncdm"])]
        if cfg.get("m_ncdm") is not None:
            args += ["--m-ncdm", str(cfg["m_ncdm"])]
        if cfg.get("T_ncdm") is not None:
            args += ["--T-ncdm", str(cfg["T_ncdm"])]
        if cfg.get("deg_ncdm") is not None:
            args += ["--deg-ncdm", str(cfg["deg_ncdm"])]
    if cfg.get("plot", True):
        args.append("--plot")
    return args


# ----------------------- Subprocess helper --------------------------

def run_subprocess(argv: List[str]) -> int:
    print("\n[agent] Launching pipeline:\n ", " ".join(argv), "\n")
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    try:
        for line in proc.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        proc.terminate()
        raise
    finally:
        proc.wait()
    return proc.returncode


# ------------------------- Chat loop -------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Chat bot that can run MG5 and CLASS pipelines via function-calling, and query a local physics KB."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Chat model (OpenAI)")
    ap.add_argument("--mg5-path", required=False,
                    help="Default path to mg5_aMC (can be provided in the chat too)")
    ap.add_argument("--mg5-pipeline-file", default=DEFAULT_MG5_PIPELINE,
                    help="Path to MG5 pipeline Python script")
    ap.add_argument("--class-pipeline-file", default=DEFAULT_CLASS_PIPELINE,
                    help="Path to CLASS pipeline Python script")
    ap.add_argument("--kb-index", default=None,
                    help="Path to local KB (kb_out) with manifest.json/chunks.jsonl/embeddings.npy/faiss.index")
    args = ap.parse_args()

    # keep kb path handy for the kb_search tool
    kb_index = args.kb_index

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("[ERROR] OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a helpful simulation orchestrator. "
        "When the user asks to run MadGraph (MG5) or CLASS, collect parameters "
        "and call the proper tool with a concise, strictly-typed JSON. "
        "Prefer sensible defaults if the user is vague. "
        "Do NOT hallucinate file paths; if not provided, ask. "
        "Only call ONE tool per message unless the user clearly asks for multiple runs. "
        "Use kb_search for physics questions when helpful; summarize with citations (title + arXiv id) from the returned chunks."
    )

    tools = build_tools_schema()
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    print_banner()

    print("[ArgoLOOM] Ready. Type your request (Ctrl+C to exit).\n")
    while True:
        try:
            user = input("> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break

        messages.append({"role": "user", "content": user})

        # ---- MODEL TURN ----
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=tools,
        )
        ai = resp.choices[0].message

        # Build assistant message WITHOUT empty tool_calls
        assistant_msg = {"role": "assistant", "content": ai.content or ""}
        if ai.tool_calls and len(ai.tool_calls) > 0:
            assistant_msg["tool_calls"] = ai.tool_calls
        messages.append(assistant_msg)

        # Debug visibility (optional)
        if ai.tool_calls:
            for tc in ai.tool_calls:
                print(f"[agent] tool call → {tc.function.name} {tc.function.arguments}")

        # Show any assistant text
        if ai.content:
            print("\n[assistant]:", ai.content, "\n")

        # If no tool calls, go to next user turn
        if not (ai.tool_calls and len(ai.tool_calls) > 0):
            continue

        # ---- EXECUTE TOOLS ----
        for tc in ai.tool_calls:
            fname = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fname,
                    "content": "ERROR: could not parse function.arguments as JSON."
                })
                continue

            if fname == "run_mg5_pipeline":
                # Ensure mg5_path exists (from tool args or CLI default)
                if not fargs.get("mg5_path"):
                    if args.mg5_path:
                        fargs["mg5_path"] = args.mg5_path
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": "run_mg5_pipeline",
                            "content": "ERROR: Missing mg5_path. Provide the full path to mg5_aMC."
                        })
                        continue
                argv = cfg_to_argv_mg5(fargs, args.mg5_pipeline_file)
                rc = run_subprocess(argv)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "run_mg5_pipeline",
                    "content": f"MG5 pipeline finished with exit code {rc}. outtag='{fargs.get('outtag','')}'"
                })

            elif fname == "run_class_pipeline":
                argv = cfg_to_argv_class(fargs, args.class_pipeline_file)
                rc = run_subprocess(argv)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "run_class_pipeline",
                    "content": f"CLASS pipeline finished with exit code {rc}. outtag='{fargs.get('outtag','')}'"
                })

            elif fname == "kb_search":
                # Do not shadow argparse 'args'
                try:
                    kargs = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": "kb_search",
                        "content": json.dumps({"error": "could not parse function.arguments as JSON"})
                    })
                    continue

                # Fill index_dir if missing: prefer tool arg, then --kb-index
                index_dir = kargs.get("index_dir") or kb_index
                if not index_dir:
                    tool_result = {"error": "kb_index not configured. Pass --kb-index or provide index_dir."}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": "kb_search",
                        "content": json.dumps(tool_result)
                    })
                    continue

                try:
                    result = toolkit_kb.kb_search(
                        index_dir=index_dir,
                        query=kargs["query"],
                        k=int(kargs.get("k", 5)),
                        engine=kargs.get("engine", "auto"),
                        doc_id=kargs.get("doc_id")
                    )
                except Exception as e:
                    result = {"error": str(e)}

                # push tool result back into the chat
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "kb_search",
                    "content": json.dumps(result),
                })

            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fname,
                    "content": f"ERROR: Unknown tool '{fname}'."
                })

        # ---- FOLLOW-UP TURN ----
        follow = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=tools,  # keep tools to allow optional chaining
        )
        follow_msg = follow.choices[0].message

        # Append follow-up WITHOUT empty tool_calls
        assistant_follow = {"role": "assistant", "content": follow_msg.content or ""}
        if follow_msg.tool_calls and len(follow_msg.tool_calls) > 0:
            assistant_follow["tool_calls"] = follow_msg.tool_calls
        messages.append(assistant_follow)

        if follow_msg.content:
            print("\n[assistant]:", follow_msg.content or "", "\n")


if __name__ == "__main__":
    main()
