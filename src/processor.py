from collections import Counter

WINDOW_SIZE = 30  # number of frames to aggregate before determining dominant expression

expression_buffer = []  # stores expressions over the current window

def processExpression(expression):
    
    expression_buffer.append(expression)

    if len(expression_buffer) >= WINDOW_SIZE:
        counts = Counter(expression_buffer)
        dominant = counts.most_common(1)[0][0]  # most frequent expression in window
        expression_buffer.clear()  # reset buffer for next window
        print(f"Dominant: {dominant}")
        return dominant

    return None