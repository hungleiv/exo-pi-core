#!/bin/bash
# Standardized harness comparison benchmark.
#
# Runs a fixed set of graded tasks against one or more exo agents (each agent
# being a different harness) and reports pass rate plus per-run mechanics:
# how many real tool calls the turn took, how many of those failed, how many
# tokens it burned, and how long it ran.
#
# Grading is objective: each case declares a regex that must match the final
# assistant message. Cases are ordered by tier, from no-tool reasoning up to
# multi-round debugging, so a harness that only breaks under long tool loops
# shows up as a tier-4 failure rather than a vague "felt worse".
#
# Usage:
#   exoharness/scripts/harness-bench.sh --agents a,b,c [--cases id,id] [--reps N]
#   exoharness/scripts/harness-bench.sh --list
#
# Results are appended as JSONL to bench-results/<timestamp>.jsonl and printed
# as a summary table at the end.

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
EXO=./target/debug/exo

# ---------------------------------------------------------------------------
# Cases: id | tier | expected-regex | prompt
# The regex is matched against the final assistant message only, so a case
# whose prompt demands a bare number at the end grades the actual answer
# rather than anything the model said while working.
# ---------------------------------------------------------------------------
CASE_IDS=(t1-arith t1-logic t2-lines t2-sum t3-counter t3-fizzbuzz t4-fixbug)

case_tier() {
  case "$1" in
    t1-*) echo "1-no-tool" ;;
    t2-*) echo "2-single-tool" ;;
    t3-*) echo "3-multi-round" ;;
    t4-*) echo "4-debug-loop" ;;
  esac
}

case_expect() {
  case "$1" in
    t1-arith)    echo '(^|[^0-9])391([^0-9]|$)' ;;
    t1-logic)    echo '[Yy]es' ;;
    t2-lines)    echo '(^|[^0-9])7([^0-9]|$)' ;;
    t2-sum)      echo '(^|[^0-9])5050([^0-9]|$)' ;;
    t3-counter)  echo '(^|[^0-9])5([^0-9]|$)' ;;
    t3-fizzbuzz) echo '(^|[^0-9])5([^0-9]|$)' ;;
    t4-fixbug)   echo '(^|[^0-9])6([^0-9]|$)' ;;
  esac
}

case_prompt() {
  case "$1" in
    t1-arith)
      echo "Compute 17 * 23. Reply with ONLY the number, nothing else." ;;
    t1-logic)
      echo "All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops Lazzies? Reply with ONLY the single word yes or no." ;;
    t2-lines)
      echo "Create /tmp/bx/lines.txt containing exactly 7 lines, where line N is the text line followed by N (line1 through line7). Then count the lines with wc -l. Finish your reply with the count as a bare number on its own line." ;;
    t2-sum)
      echo "Use one shell command to compute the sum of the integers 1 through 100 (for example with seq and awk, or bash arithmetic). Finish your reply with the resulting sum as a bare number on its own line." ;;
    t3-counter)
      echo "In /tmp/bx, create counter.txt containing 0. Then, one shell call at a time and never combining steps, read the current number, add exactly 1, and write it back. Do that single-increment cycle 5 separate times as 5 separate tool calls. Then cat counter.txt and finish your reply with the final number as a bare number on its own line." ;;
    t3-fizzbuzz)
      echo "Write /tmp/bx/fizz.sh that prints FizzBuzz for the numbers 1 to 15 (Fizz for multiples of 3, Buzz for multiples of 5, FizzBuzz for both, otherwise the number). Run it, then use grep -c to count how many of its output lines contain the word Fizz. Finish your reply with that count as a bare number on its own line." ;;
    t4-fixbug)
      printf '%s' "Create /tmp/bx/buggy.sh with exactly this content:
#!/bin/bash
total=0
for i in 1 2 3; do
  total=\$((total + i)
done
echo \$total

Then run it with bash. It contains a deliberate syntax error. Read the error output, fix the script, re-run it until it works, and finish your reply with the number it finally prints, as a bare number on its own line." ;;
  esac
}

# ---------------------------------------------------------------------------
AGENTS=""
CASES=""
REPS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agents) AGENTS="$2"; shift 2 ;;
    --cases)  CASES="$2"; shift 2 ;;
    --reps)   REPS="$2"; shift 2 ;;
    --list)
      for c in "${CASE_IDS[@]}"; do
        printf '%-12s tier=%-14s expect=/%s/\n' "$c" "$(case_tier "$c")" "$(case_expect "$c")"
      done
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$AGENTS" ]]; then
  echo "usage: $0 --agents <slug,slug,...> [--cases id,id] [--reps N]" >&2
  exit 1
fi
[[ -z "$CASES" ]] && CASES="$(IFS=,; echo "${CASE_IDS[*]}")"

mkdir -p bench-results
OUT="bench-results/$(date +%Y%m%d-%H%M%S).jsonl"

# Pull the graded answer and the run mechanics out of the conversation's event
# log. Both harness families are covered: exo-native tool results carry
# .ok, pi's carry .is_error.
collect() {
  local agent="$1" slug="$2"
  $EXO conversation events "$agent" "$slug" 2>/dev/null | jq -c '
    def texts: [.events[] | select(.data.type=="messages") | .data.messages[]
                | select(.role=="assistant")
                | if (.content|type)=="string" then .content
                  else ([.content[]? | select(.type=="text") | .text] | join(""))
                  end];
    {
      final: (texts | map(select(length > 0)) | last // ""),
      rounds: ([.events[] | select(.data.type=="tool_requested")] | length),
      tool_errors: ([.events[] | select(.data.type=="tool_result")
                     | select((.data.result.ok? == false) or (.data.result.is_error? == true))] | length),
      prompt_tokens: ([.events[] | select(.data.type=="messages") | .data.usage.prompt_tokens? // 0] | add // 0),
      completion_tokens: ([.events[] | select(.data.type=="messages") | .data.usage.completion_tokens? // 0] | add // 0)
    }'
}

IFS=',' read -ra AGENT_LIST <<< "$AGENTS"
IFS=',' read -ra CASE_LIST <<< "$CASES"

for agent in "${AGENT_LIST[@]}"; do
  for case_id in "${CASE_LIST[@]}"; do
    expect="$(case_expect "$case_id")"
    prompt="$(case_prompt "$case_id")"
    [[ -z "$expect" ]] && { echo "unknown case: $case_id" >&2; continue; }
    for ((rep = 1; rep <= REPS; rep++)); do
      slug="bench-${case_id}-r${rep}"
      $EXO conversation delete "$agent" "$slug" >/dev/null 2>&1
      $EXO conversation create "$agent" "$slug" >/dev/null 2>&1

      # %3N is not honoured everywhere, so take nanoseconds and scale down.
      start=$(( $(date +%s%N) / 1000000 ))
      $EXO conversation send "$agent" "$slug" "$prompt" >/dev/null 2>&1
      end=$(( $(date +%s%N) / 1000000 ))

      metrics="$(collect "$agent" "$slug")"
      [[ -z "$metrics" ]] && metrics='{"final":"","rounds":0,"tool_errors":0,"prompt_tokens":0,"completion_tokens":0}'
      final="$(echo "$metrics" | jq -r '.final')"

      if echo "$final" | grep -Eq "$expect"; then pass=true; else pass=false; fi

      echo "$metrics" | jq -c \
        --arg agent "$agent" --arg case "$case_id" --arg tier "$(case_tier "$case_id")" \
        --argjson rep "$rep" --argjson pass "$pass" --argjson ms "$((end - start))" \
        '{agent: $agent, case: $case, tier: $tier, rep: $rep, pass: $pass, ms: $ms,
          rounds, tool_errors, prompt_tokens, completion_tokens,
          final: (.final | gsub("\\s+"; " ") | .[0:120])}' >> "$OUT"

      printf '%-16s %-12s rep%-2s pass=%-5s rounds=%-3s errs=%-3s %ss\n' \
        "$agent" "$case_id" "$rep" "$pass" \
        "$(echo "$metrics" | jq -r '.rounds')" \
        "$(echo "$metrics" | jq -r '.tool_errors')" \
        "$(( (end - start) / 1000 ))"
    done
  done
done

echo
echo "=== summary (per agent x tier) ==="
jq -s -r '
  group_by(.agent + "|" + .tier)[]
  | {agent: .[0].agent, tier: .[0].tier,
     runs: length,
     passed: ([.[] | select(.pass)] | length),
     avg_rounds: (([.[] | .rounds] | add) / length),
     tool_errors: ([.[] | .tool_errors] | add),
     avg_s: ((([.[] | .ms] | add) / length / 1000) | floor)}
  | "\(.agent | .[0:16] | . + "                " | .[0:16]) \(.tier | . + "              " | .[0:14]) pass \(.passed)/\(.runs)  avg_rounds \(.avg_rounds)  tool_errors \(.tool_errors)  avg \(.avg_s)s"
' "$OUT"

echo
echo "results: $OUT"
