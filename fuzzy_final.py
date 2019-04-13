def fuzzy(a,b,c):
    a1=''
    a2=''
    a3=''
    if(a>0 and a<0.34):
        a1='low'
    elif(a>0.34 and a<0.68):
        a1='medium'
    else:
        a1='high'
    if(b>0 and b<0.34):
        b1='low'
    elif(b>0.34 and b<0.68):
        b1='medium'
    else:
        b1='high'
    if(c==0):
        c1='low'
    else:
        c1='high'
    print(a1=='medium')
    if(a1=='low' and b1=='low' and c1=='low'):
        return ('very low')
    if(a1=='low' and b1=='low' and c1=='high'):
        return ('low')
    if(a1=='low' and b1=='medium' and c1=='low'):
        return('low')
    if(a1=='low' and b1=='high' and c1=='low'):
        return('low')
    if(a1=='low' and b1=='medium' and c1=='high'):
        return('low')
    if(a1=='low' and b1=='high' and c1=='high'):
        return('medium')
    if(a1=='medium' and b1=='low' and c1=='low'):
        return('low')
    if(a1=='medium' and b1=='low' and c1=='high'):
        return('medium')
    if(a1=='medium' and b1=='medium' and c1=='high'):
        return('medium')
    if(a1=='medium' and b1=='medium' and c1=='low'):
        return('medium')
    if(a1=='medium' and b1=='high' and c1=='low'):
        return('medium')
    if(a1=='medium' and b1=='high' and c1=='high'):
        return('high')
    if(a1=='high' and b1=='low' and c1=='low'):
        return('medium')
    if(a1=='high' and b1=='low' and c1=='high'):
        return('high')
    if(a1=='high' and b1=='medium' and c1=='high'):
        return('high')
    if(a1=='high' and b1=='medium' and c1=='low'):
        return('high')
    if(a1=='high' and b1=='high' and c1=='low'):
        return('high')
    if(a1=='high' and b1=='high' and c1=='high'):
        return('very high')

print(fuzzy(0.4,0.6,1))