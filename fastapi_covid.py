import subprocess
from fastapi import FastAPI

app = FastAPI()

##########################################################
# Launch a command with pipes
p = subprocess.Popen(['python examples/interactive.py -mf model/poly/covid7 --single-turn True'], shell=True,
                     stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)

while 1:
    line = p.stdout.readline()
    line = line.strip().decode()
    print(line)
    if line == '[  polyencoder_type: codes ]':
        break
#the model is ready for interactivation
##########################################################
@app.get("/")
def root(question: str = "What is Covid-19?"):
    # Send the question and get the output
    p.stdin.write(bytes(question, 'utf-8'))
    p.stdin.write(bytes("\n", 'utf-8'))
    p.stdin.flush()
    line=''
    while '[Polyencoder]' not in line: # Exclude warnings and other messages
        line = p.stdout.readline()
        line = line.strip().decode()  # To interpret as text, decode
    line = line.split('[Polyencoder]:')
    print(line)
    result=line[-1].strip()
    return {"question": question, "answers": result}