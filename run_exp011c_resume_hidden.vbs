' Exp 011c resume - hidden user-session launcher.
' Mirrors the Hermes_APIServer.vbs pattern: window style 0 = hidden console,
' completion not waited on, so the trainer is NOT a child of the Hermes app.
Option Explicit
Dim sh
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "D:\Projects\llm-lab"
sh.Run "cmd /c D:\Projects\llm-lab\run_exp011c_resume.bat", 0, False
