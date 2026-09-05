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
CASE_IDS=(t1-arith t1-logic t2-lines t2-sum t3-counter t3-fizzbuzz t4-fixbug t5-multifile t5-pipeline t6-refactor t6-twobugs t6-report)

case_tier() {
  case "$1" in
    t1-*) echo "1-no-tool" ;;
    t2-*) echo "2-single-tool" ;;
    t3-*) echo "3-multi-round" ;;
    t4-*) echo "4-debug-loop" ;;
    t5-*) echo "5-long-multistep" ;;
    # Tier 6 exists because tier 5 stopped separating the harnesses once file
    # tools landed: several agents sat at 6/6 there. These run longer (more
    # files, more edits to existing content, more chances for a quoting or
    # stale-read mistake to compound) rather than merely asking for more steps.
    t6-*) echo "6-hard-multifile" ;;
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
    t5-multifile) echo '(^|[^0-9])3/3([^0-9]|$)' ;;
    t5-pipeline)  echo '(^|[^0-9])10([^0-9]|$)' ;;
    t6-refactor)  echo '(^|[^0-9])9([^0-9]|$)' ;;
    t6-twobugs)   echo '(^|[^0-9])30([^0-9]|$)' ;;
    t6-report)    echo '(^|[^0-9])3([^0-9]|$)' ;;
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
      echo "Write /tmp/bx/fizz.sh that prints FizzBuzz for the numbers 1 to 15 (Fizz for multiples of 3, Buzz for multiples of 5, FizzBuzz for both, otherwise the number). Run it, then use grep -c to count how many of its output lines contain Fizz as a substring - a FizzBuzz line counts as one of them. Finish your reply with that count as a bare number on its own line." ;;
    t4-fixbug)
      printf '%s' "Create /tmp/bx/buggy.sh with exactly this content:
#!/bin/bash
total=0
for i in 1 2 3; do
  total=\$((total + i)
done
echo \$total

Then run it with bash. It contains a deliberate syntax error. Read the error output, fix the script, re-run it until it works, and finish your reply with the number it finally prints, as a bare number on its own line." ;;
    t5-multifile)
      echo "In /tmp/proj, build a tiny bash calculator library across two files: lib.sh defining three functions add(), sub(), mul() (each takes two args and echoes the arithmetic result using \$((...))), and test.sh that sources lib.sh and runs exactly 3 checks: add 2 3 should print 5, sub 10 4 should print 6, mul 3 3 should print 9. test.sh should print PASS or FAIL per check and a final summary line in the exact form 'N/3 passed'. Run test.sh. If any check fails, find and fix the bug (it may be in lib.sh or in test.sh's expected values), then re-run test.sh until it reports 3/3 passed. Finish your reply with that exact final summary line, nothing else added after it." ;;
    t5-pipeline)
      echo "In /tmp/proj2, build a 3-stage bash pipeline as 3 separate scripts, running each one with a separate tool call in order: gen.sh writes the integers 1 through 20, one per line, to raw.txt. filter.sh reads raw.txt and writes only the even numbers, one per line, to filtered.txt. count.sh reads filtered.txt and writes the number of lines in it to result.txt. Run gen.sh, then filter.sh, then count.sh, each as its own tool call - do not combine them into one command. Then cat result.txt. Finish your reply with the number from result.txt as a bare number on its own line." ;;
    t6-refactor)
      echo "In /tmp/t6a, create lib.sh defining a bash function named add_all that sums all of its arguments and echoes the total, and main.sh that sources ./lib.sh and calls add_all 2 3 4, echoing the result. Run main.sh with bash and confirm it prints 9. Then rename the function from add_all to sum_all in BOTH files, so no reference to add_all remains anywhere. Run main.sh again and confirm it still works. Finish your reply with the number main.sh prints after the rename, as a bare number on its own line." ;;
    t6-twobugs)
      printf '%s' "Create /tmp/t6b/buggy.sh with exactly this content:
#!/bin/bash
total=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  if [ \$((i % 2)) -eq 1 ]; then
    total=\$((total + i)
  fi
done
echo \$total

This script is meant to sum the EVEN numbers from 1 to 10 (2+4+6+8+10), but it has two deliberate bugs: one syntax error that stops it running at all, and one logic error that makes it sum the wrong numbers. Run it, read the error, fix both bugs, and re-run until it prints the correct even-number total. Finish your reply with the number it finally prints, as a bare number on its own line." ;;
    t6-report)
      echo "In /tmp/t6c/data, create five files f1.csv through f5.csv containing exactly 2, 4, 6, 1 and 5 lines respectively (any text content, one record per line). Then write /tmp/t6c/count.sh which examines every .csv file in /tmp/t6c/data and echoes how many of them have MORE THAN 3 lines. Run count.sh. Then verify by listing the line count of each file. Finish your reply with the number count.sh printed, as a bare number on its own line." ;;
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
      completion_tokens: ([.events[] | select(.data.type=="messages") | .data.usage.completion_tokens? // 0] | add // 0),
      nudges: ([.events[] | select(.data.type=="artifact_written")
                | select((.data.path? // "") | startswith("pi-core/nudge-"))] | length)
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
      [[ -z "$metrics" ]] && metrics='{"final":"","rounds":0,"tool_errors":0,"prompt_tokens":0,"completion_tokens":0,"nudges":0}'
      final="$(echo "$metrics" | jq -r '.final')"

      if echo "$final" | grep -Eq "$expect"; then pass=true; else pass=false; fi

      echo "$metrics" | jq -c \
        --arg agent "$agent" --arg case "$case_id" --arg tier "$(case_tier "$case_id")" \
        --argjson rep "$rep" --argjson pass "$pass" --argjson ms "$((end - start))" \
        '{agent: $agent, case: $case, tier: $tier, rep: $rep, pass: $pass, ms: $ms,
          rounds, tool_errors, prompt_tokens, completion_tokens, nudges,
          final: (.final | gsub("\\s+"; " ") | .[0:120])}' >> "$OUT"

      printf '%-18s %-12s rep%-2s pass=%-5s rounds=%-3s errs=%-3s nudges=%-3s %ss\n' \
        "$agent" "$case_id" "$rep" "$pass" \
        "$(echo "$metrics" | jq -r '.rounds')" \
        "$(echo "$metrics" | jq -r '.tool_errors')" \
        "$(echo "$metrics" | jq -r '.nudges')" \
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
     nudges: ([.[] | .nudges // 0] | add),
     avg_s: ((([.[] | .ms] | add) / length / 1000) | floor)}
  | "\(.agent | .[0:18] | . + "                  " | .[0:18]) \(.tier | . + "              " | .[0:14]) pass \(.passed)/\(.runs)  avg_rounds \(.avg_rounds)  tool_errors \(.tool_errors)  nudges \(.nudges)  avg \(.avg_s)s"
' "$OUT"

echo
echo "results: $OUT"
