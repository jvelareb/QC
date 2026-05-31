$py = conda run -n py_all python -c "import sys; print(sys.executable)"
mkdir .vscode -Force | Out-Null
@"
{
  "python.defaultInterpreterPath": "$py"
}
"@ | Set-Content -Encoding UTF8 .vscode\settings.json
