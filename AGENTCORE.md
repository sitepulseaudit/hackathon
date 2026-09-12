# Deploy LLMCite on Amazon Bedrock AgentCore

## Prerequisites
- AWS account (`sitepulseaudit`) with Bedrock model access in `us-east-1`
- Python 3.10+

## Configure + deploy
```bash
source .venv/bin/activate
pip install -r requirements.txt bedrock-agentcore bedrock-agentcore-starter-toolkit
agentcore configure --entrypoint runtime_app.py --name llmcite --region us-east-1
agentcore launch
```

AgentCore strengthens the hackathon Technological Implementation score; optional but recommended.


## Deployed instance

- ARN: `arn:aws:bedrock-agentcore:us-east-1:633398616029:runtime/llmcite-iZWSrK5CJ7`
- Endpoint: DEFAULT
- Account: `633398616029`


## Live vs mock

The public checkout form sends `brand` + `product_url` with `backend: live`. AgentCore then:

1. Fetches the URL and writes `active_pack.json`
2. Asks Bedrock to generate buyer queries + competitors (template fallback if JSON fails)
3. Probes Bedrock for consumer answers (does **not** tell the model to name the brand)
4. Scores + drafts a citation page for **that** site

Local `python agent.py --mode pipeline` stays on SleepFix **mock** fixtures.

## IAM

The **AgentCore runtime role** (not the Lambda API role) must allow:

```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
  "Resource": "*"
}
```

Enable model access in `us-east-1` for `amazon.nova-lite-v1:0` (or `amazon.nova-micro-v1:0`). Optional env on the runtime (and Lambda, forwarded in the payload): `LLMCITE_BEDROCK_MODEL_ID`. Lambda timeout is 120s because live packs issue several Bedrock calls.

## Web form without AgentCore endpoint

The public checkout form (`lambda_api`) defaults to **`LLMCITE_INVOKE_MODE=local`**: it imports `agent.py` inside Lambda and runs `build_pack` → `probe_queries(live)` → `score_visibility` → `draft_citation_page` against Bedrock (`amazon.nova-lite-v1:0`). This bypasses AgentCore because `Create/List/Get AgentRuntimeEndpoint` is AccessDenied even for account root.

- Redeploy API: `cd lambda_api && ./deploy.sh`
- Optional AgentCore path: set `LLMCITE_INVOKE_MODE=agentcore` (keeps ARN + DEFAULT qualifier) once endpoint IAM is restored.
- AgentCore runtime above remains deployed for the hackathon.
