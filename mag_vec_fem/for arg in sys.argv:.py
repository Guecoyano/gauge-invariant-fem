for arg in sys.argv:
    if arg.startwith("N_eig"):
        N_eig = eval(arg.replace("N_eig", ""))
    elif arg.startwith("lnm"):
        lnm = eval(arg.replace("lnm", ""))
    elif arg.startwith("B"):
        B = eval(arg.replace("B", ""))
    elif arg.startwith("V"):
        V_maxmeV = eval(arg.replace("V_maxmeV", ""))
    elif arg.startwith("h"):
        h = eval(arg.replace("h", ""))
    elif arg.startwith("pot"):
        pot_version = eval(arg.replace("pot_version", ""))
    elif arg.startwith("gauge"):
        gauge = arg.replace("gauge", "")
