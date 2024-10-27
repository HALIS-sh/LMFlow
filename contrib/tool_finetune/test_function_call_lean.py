import subprocess

lean_file = '/home/wenhesun/Lean_env/MyLeanExample/Main.lean'

# 调用 Lean 脚本
result = subprocess.run(['lean', '--run', lean_file], capture_output=True, text=True)

# 处理输出
if result.returncode == 0:
    output = result.stdout.strip()
    print("Lean 输出:")
    print(output)

else:
    print("Lean 执行出错:")
    print(result.stderr)
