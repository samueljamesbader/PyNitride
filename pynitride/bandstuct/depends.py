
def varshni(material):
    T=material._pmdb["T"]
    return material["varshni.Eg0"]-material["varshni.alpha"]*T**2/(T+material["varshni.beta"])

