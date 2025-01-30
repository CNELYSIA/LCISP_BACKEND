def userProcess(image):
    qa = image.select('QA60')
    # Bits 10 and 11是云，我们要把它mask掉
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 1
    # 这两个标志都应该设置为0，表示条件明确。
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    # 哨兵的像元值是反射率的10000倍，要除以10000
    return image.updateMask(mask).divide(10000)