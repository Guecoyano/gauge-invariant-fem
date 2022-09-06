#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 17:19:47 2021

@author: alioune Seye
"""
'''on veut résoudre -delta u + (C+lambda/r^2)u=1 en dimension 2 en développant autour de r=1'''
import numpy as np

def der(l):
    for i in range(len(l)):
        l[i]=i*l[i]
    l=l[1:]
    return l
    
def unsurR(n):#liste de n+1 termes, soit développement en x^n compris
    l=[]
    for i in range(n+1):
        l+=[(-1)**i]
    return l
    
def unsurR2(n):
    l=[]
    for i in range(n+1):
        l+=[(-1)**i*(i+1)]
    return l

def mult(l,m):
    p=[]
    for i in range(min(len(l),len(m))):
        p+=[0]
        for k in range(i+1):
            p[i]+=l[k]*m[i-k]
    return p

def one(n):
    o=[1]
    if n==0:
        return o
    else:
        for i in range(n):
            o+=[0]
        return o
    
def expansion(C,l,n):