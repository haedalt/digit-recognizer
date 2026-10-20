@echo off
rem Launch the digit recognizer without keeping a console window open.
rem pythonw runs the GUI with no console; "start" returns immediately so this window closes.
start "" pythonw "%~dp0digit_recognizer.py"
