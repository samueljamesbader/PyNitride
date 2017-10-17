import numpy as np
import re


def read(filename):
    with open(filename) as f:
        problems=[]
        glob={}
        state="outside problem"
        for line in f:
            if state=="outside problem":
                if line.strip()=="Starting solve of next problem:":
                    problem={}
                    problems+=[problem]
                    attempts=[]
                    problem['attempts']=attempts
                    problemstatement=""
                    for line in f:
                        if line.startswith("====="): break
                        problemstatement+=line
                    problem['statement']=problemstatement
                    #print("Starting problem:  " + problem['statement'])
                    state="outside iteration"
                    continue
                mo=re.match(r"wallclock: (\d\.eE\+\-) s",line)
                if mo: problem['wallclock']=float(mo.group(1))
                mo=re.match(r"total cpu: (\d\.eE\+\-) s",line)
                if mo: problem['total cpu']=float(mo.group(1))
            if state=="outside iteration":
                #print("###"+line)
                #if line.startswith("Computing step"):
                    #print("          "+line)
                mo=re.match(
                    r"^Computing step from t=(?P<start>[\d\.eE\+\-]+) to t=(?P<end>[\d\.eE\+\-]+)" \
                    " \(Stepsize: (?P<step>[\d\.eE\+\-]+)\) :",
                    line)
                if mo:
                    iteration={k:float(v) for k,v in mo.groupdict().items()}
                    state="inside iteration"
                    #print("Entering iteration")
                    continue
                if line.strip()=="Finished, because of...":
                    line=next(f).strip()
                    problem['reason']=line
                    problem['success']=(line=="Curve trace finished.")
                    state="outside problem"
                    continue
                if line.strip()=="done.":
                    state="outside problem"
                    continue
            elif state=="inside iteration":
                if line.strip()=="Finished, because...":
                    line=next(f).strip()
                    iteration['reason']=line
                    iteration['success']=line.startswith("Error smaller") or line.startswith("RHS smaller")
                    for line in f:
                        mo=re.match(r"Total time:\s+([\d\.eE\+\-]+) s",line)
                        if mo:
                            iteration['total time']=float(mo.group(1))
                            break
                    if iteration['success']:
                        for line in f:
                            if line.startswith("contact"):
                                iteration['result']={}
                                for line in f:
                                    mo=re.match(r"^ (?P<contact>[\w]+)\s+(?P<voltage>[\d\.eE\+\-]+)\s+" \
                                                "(?P<ecurrent>[\d\.eE\+\-]+)\s+(?P<hcurrent>[\d\.eE\+\-]+)\s+" \
                                                "(?P<current>[\d\.eE\+\-]+)\s+",
                                                line)
                                    if mo: iteration['result'][mo.groupdict()['contact']]=mo.groupdict()
                                    else: break
                                break
                            if line.startswith("Contact"):
                                iteration['result']={}
                                next(f)
                                for line in f:
                                    mo=re.match(r"^ (?P<contact>[\w]+)\s+(?P<voltage>[\d\.eE\+\-]+)\s+(?P<innervoltage>[\d\.eE\+\-]+)\s+" \
                                                "(?P<ecurrent>[\d\.eE\+\-]+)\s+(?P<hcurrent>[\d\.eE\+\-]+)\s+" \
                                                "(?P<current>[\d\.eE\+\-]+)\s+",
                                                line)
                                    if mo: iteration['result'][mo.groupdict()['contact']]=mo.groupdict()
                                    else: break
                                break
                    attempts+=[iteration]
                    state="outside iteration"
                    #print("finished iter on: "+line)
        return problems

