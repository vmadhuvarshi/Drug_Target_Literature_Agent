param (
    [Parameter(ValueFromRemainingArguments=$true)]
    $RemainingArgs
)

# Run the benchmark script passing standard arguments
python -m eval.benchmark $RemainingArgs
