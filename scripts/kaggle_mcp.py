"""Minimal client for Kaggle's MCP endpoint (streamable HTTP, JSON-RPC).

Token: $KAGGLE_API_TOKEN, else ~/.kaggle/mcp_token (never printed, never committed).
Usage:
    python3 -I scripts/kaggle_mcp.py list
    python3 -I scripts/kaggle_mcp.py schema TOOL_NAME
    python3 -I scripts/kaggle_mcp.py call TOOL_NAME '{"request": {...}}'
"""
import json
import os
import sys
import urllib.request

URL = "https://www.kaggle.com/mcp"
TOKEN = os.environ.get("KAGGLE_API_TOKEN") or open(os.path.expanduser("~/.kaggle/mcp_token")).read().strip()


class Client:
    def __init__(self):
        self.session = None
        self.id = 0

    def rpc(self, method, params=None, notify=False):
        body = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        if not notify:
            self.id += 1
            body["id"] = self.id
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST")
        req.add_header("Authorization", f"Bearer {TOKEN}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        if self.session:
            req.add_header("Mcp-Session-Id", self.session)
        with urllib.request.urlopen(req, timeout=300) as r:
            sid = r.headers.get("Mcp-Session-Id")
            if sid:
                self.session = sid
            raw = r.read().decode()
        if notify or not raw.strip():
            return None
        if raw.lstrip().startswith("{"):
            return json.loads(raw)
        # SSE: take the last data line carrying a JSON-RPC response
        out = None
        for line in raw.splitlines():
            if line.startswith("data:"):
                try:
                    out = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    pass
        return out

    def start(self):
        r = self.rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                    "clientInfo": {"name": "sqk", "version": "0.1"}})
        self.rpc("notifications/initialized", notify=True)
        return r


def main():
    c = Client()
    c.start()
    if sys.argv[1] == "list":
        r = c.rpc("tools/list")
        for t in r["result"]["tools"]:
            print(t["name"], "::", (t.get("description") or "").split("\n")[0][:150])
    elif sys.argv[1] == "schema":
        r = c.rpc("tools/list")
        for t in r["result"]["tools"]:
            if t["name"] == sys.argv[2]:
                print(json.dumps(t, indent=1)[:4000])
    elif sys.argv[1] == "call":
        args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        r = c.rpc("tools/call", {"name": sys.argv[2], "arguments": args})
        print(json.dumps(r, indent=1)[:6000])


if __name__ == "__main__":
    main()
