import subprocess
from fastapi import FastAPI

app = FastAPI()

p = subprocess.Popen(['python -m parlai.scripts.interactive -mf ../model/poly/covid7'], shell=True,
                     stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)

while 1:
    line = p.stdout.readline()
    line = line.strip().decode()
    #print(line)
    if line == '[  polyencoder_type: codes ]':
        break

@app.get("/")
def root(question: str = "What is Covid-19?"):
    # Send the data and get the output
    p.stdin.write(bytes(question, 'utf-8'))
    p.stdin.write(bytes("\n", 'utf-8'))
    p.stdin.flush()
    # To interpret as text, decode
    line = p.stdout.readline()
    # p.stdout.flush()
    line = line.strip().decode()

    if '[Polyencoder]' in line:
        line = line.split('[Polyencoder]:')
    print(line)
    result=line[-1]
    return {"question": question, "answers": result}