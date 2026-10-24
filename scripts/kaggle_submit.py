"""Upload a submission CSV to Kaggle and submit it, then wait for the score.

    python3 -I scripts/kaggle_submit.py data/submission.csv "description"

Run scripts/validate.py first; only submit files that pass in both modes.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_mcp import Client  # noqa: E402

COMP = "squares-in-squares"


def call(c, tool, req):
    r = c.rpc("tools/call", {"name": tool, "arguments": {"request": req}})
    return json.loads(r["result"]["content"][0]["text"])


def main():
    path, desc = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
    c = Client()
    c.start()
    up = call(c, "start_competition_submission_upload", {
        "competitionName": COMP, "hasCompetitionName": True, "contentLength": os.path.getsize(path),
        "lastModifiedEpochSeconds": int(os.path.getmtime(path)), "fileName": os.path.basename(path)})
    data = open(path, "rb").read()
    req = urllib.request.Request(up["create_url"], data=data, method="PUT")
    req.add_header("Content-Type", "application/octet-stream")
    urllib.request.urlopen(req, timeout=900).read()
    sub = call(c, "submit_to_competition", {"competitionName": COMP, "blobFileTokens": up["token"],
                                            "submissionDescription": desc, "hasSubmissionDescription": bool(desc)})
    ref = sub["ref"]
    print("submitted, ref", ref, flush=True)
    for _ in range(120):
        time.sleep(30)
        s = call(c, "get_competition_submission", {"ref": ref})
        if any(k in s for k in ("public_score", "status", "error_description")):
            print(json.dumps(s, indent=1))
            return
    print("still scoring; check later with get_competition_submission ref", ref)


if __name__ == "__main__":
    main()
