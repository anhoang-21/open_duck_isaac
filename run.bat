@echo off
:: Run a script with the Isaac Sim 6.2 / Isaac Lab 3.0 python (IsaacLab\env_isaaclab).
::   run.bat scripts\convert_usd.py
::   run.bat scripts\train.py --num-envs 16384
set ISAAC=C:\Users\ADMIN\Isaacsim\IsaacLab\env_isaaclab\Scripts\python.exe
set OMNI_KIT_ACCEPT_EULA=YES
set PATH=C:\Program Files\Git\cmd;%PATH%
cd /d %~dp0
"%ISAAC%" %*
