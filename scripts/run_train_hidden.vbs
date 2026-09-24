' Training launch - hidden user-session launcher.
' Mirrors the Hermes_APIServer.vbs / run_exp011c_resume_hidden.vbs pattern:
' window style 0 = no console, completion not waited on, so the trainer is a child of
' the Task Scheduler service rather than of the Hermes app.
Option Explicit
Dim sh, extra
Set sh = CreateObject("WScript.Shell")
If WScript.Arguments.Count > 0 Then extra = WScript.Arguments(0) Else extra = ""
sh.CurrentDirectory = "D:\Projects\llm-lab"
sh.Run "cmd /c D:\Projects\llm-lab\run_train.bat " & extra, 0, False
