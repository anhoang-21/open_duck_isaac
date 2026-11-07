@echo off
:: Watch the trained policy walk in the Isaac Sim window (double-click, or run with options):
::   play.bat                         newest checkpoint in logs\run1, 16 robots, random commands
::   play.bat --num-envs 64 --vx 0.1  64 robots walking forward at 0.1 m/s
::   play.bat --mpc-only --vx 0.1     the MPC walker alone, for comparison
::   play.bat --checkpoint logs\run1\model_1649.pt --no-dr
set ISAAC=C:\Users\ADMIN\Isaacsim\IsaacLab\env_isaaclab\Scripts\python.exe
set OMNI_KIT_ACCEPT_EULA=YES
cd /d %~dp0
"%ISAAC%" scripts\play.py %*
