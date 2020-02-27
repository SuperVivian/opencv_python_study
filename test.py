if __name__ == '__main__':
    # 【1】初始化
    a = []
    b = [1, 'hello', [1], False]
    list_c = list('hello')
    # 【2】获取数组长度
    print("list c的长度为：%d" % len(list_c))
    # 【3】访问元素
    d = list_c[-2]
    e = list_c[1:2]
    f = list_c[:]
    # 【4】迭代所有元素
    print("[Version 1] The contents of a1 are:")
    for c in list_c:
        print(c)
    print("[Version 2] The contents of a1 are:")
    for i, c in enumerate(list_c):
        print('第%d个元素是%c' % (i, c))
    # 【5】修改/添加元素
    list_c[0] = '1'
    list_c.append('Runoob')
    list_c.insert(0, '1')
    list_c.extend(b)
    # 【6】删除/清空元素
    del list_c[2]
    list_c.pop()
    list_c.remove('1')
    list_c = []
    # 【7】查询某个值的索引、查询某个值出现次数
    print(list_c.index('1'))
    print(list.count('g'))
    # 【8】组合列表
    list_d = [1, 2, 3] + [4, 5, 6]
    # 【9】重复列表
    list_e = [1, 2, 3] * 4
    # 【10】判断元素是否在列表中
    print(5 in list_e)
    # 【11】排序
    list_c.sort(reverse=True)
    # 【12】逆序
    list_c.reverse()
