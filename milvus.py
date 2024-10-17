from pymilvus import connections

connections.connect("default", host='localhost', port='19530')
print("Connected to Milvus")
