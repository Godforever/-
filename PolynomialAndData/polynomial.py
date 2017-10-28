def polynomial(x, w):
    answer = 0.0
    for i in range(w.size):
        answer += w[i] * pow(x, i)
    return answer