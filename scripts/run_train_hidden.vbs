' Training launch - hidden user-session launcher.
' Mirrors the Hermes_APIServer.vbs / run_exp011c_resume_hidden.vbs pattern:
' window style 0 = no console, completion not waited on, so the trainer is a child of
' the Task Scheduler service rather than of the Hermes app.
Option Explicit
Dim sh, extra
Set sh = CreateObject("WScript.Shell")
If WScript.Arguments.Count > 0 Then extra = WScript.Arguments(0) Else extra = ""
sh.CurrentDirectory = "D:\Projects\llm-lab"
' The scripts prefix is load-bearing: this file lives alongside the .bat, so a root-level
' path resolves to a file that does not exist. wscript.Run with a missing target returns
' exit 0 and writes nothing to the launch log, so the chain looks like it fired while no
' trainer ever starts - the exact failure on 2026-09-30 20:51.
' Path is assembled from Chr(92) so the separators cannot be mangled by an editing layer.
Dim B
B = Chr(92)
sh.Run "cmd /c D:" & B & "Projects" & B & "llm-lab" & B & "scripts" & B & "run_train.bat " & extra, 0, False
