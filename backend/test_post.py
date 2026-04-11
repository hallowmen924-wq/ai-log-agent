import requests
r = requests.post('http://127.0.0.1:8001/populate-sample')
print('STATUS', r.status_code)
print(r.text)
