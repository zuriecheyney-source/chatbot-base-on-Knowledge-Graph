
from py2neo import Graph, ConnectionProfile
import json

NEO4J_URI = "bolt+s://5c6eff81.databases.neo4j.io:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "0jf3l-mQo-huaIL02Qq8Hbimq1wJROZEYfQHZPToe2U"

def check():
    profile = ConnectionProfile(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), verify=False)
    g = Graph(profile)
    
    print("Checking for '发热' variants...")
    q1 = "MATCH (n:Symptom) WHERE n.name = '发热' RETURN n.name"
    print(f"Exact '发热': {g.run(q1).data()}")
    
    q3 = "MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) WHERE n.name CONTAINS '发热' RETURN m.name, r.name, n.name LIMIT 5"
    print(f"Full Disease-Symptom for '发热': {g.run(q3).data()}")

if __name__ == '__main__':
    check()
