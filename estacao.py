"""
Modelagem interativa de rede de transporte (ônibus/metrô) como grafo.
Agora com entrada via terminal para o usuário digitar estações, conexões e consultas.

Funções:
- Adicionar/remover estações.
- Adicionar/remover conexões com tempo e linha.
- Consultar rotas a partir de uma estação.
- Verificar se é possível chegar de A a B.
- Calcular menor trajeto (Dijkstra simples).
- Calcular trajeto mais rápido com baldeações (Dijkstra com espera).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set
import heapq

@dataclass
class Edge:
    to_station: str
    time: float
    line: Optional[str] = None

class TransportGraph:
    def __init__(self, default_transfer_wait: float = 4.0):
        self.adj: Dict[str, List[Edge]] = {}
        self.transfer_waits: Dict[str, float] = {}
        self.default_transfer_wait = float(default_transfer_wait)

    def add_station(self, station_id: str, transfer_wait: Optional[float] = None) -> None:
        if station_id not in self.adj:
            self.adj[station_id] = []
        if transfer_wait is not None:
            self.transfer_waits[station_id] = float(transfer_wait)

    def remove_station(self, station_id: str) -> None:
        if station_id not in self.adj:
            return
        del self.adj[station_id]
        for s, edges in list(self.adj.items()):
            self.adj[s] = [e for e in edges if e.to_station != station_id]
        if station_id in self.transfer_waits:
            del self.transfer_waits[station_id]

    def add_connection(self, from_station: str, to_station: str, time: float, line: Optional[str] = None, bidirectional: bool = True) -> None:
        if from_station not in self.adj:
            self.add_station(from_station)
        if to_station not in self.adj:
            self.add_station(to_station)
        self.adj[from_station].append(Edge(to_station, float(time), line))
        if bidirectional:
            self.adj[to_station].append(Edge(from_station, float(time), line))

    def remove_connection(self, from_station: str, to_station: str, line: Optional[str] = None, bidirectional: bool = True) -> None:
        if from_station in self.adj:
            self.adj[from_station] = [e for e in self.adj[from_station] if not (e.to_station == to_station and (line is None or e.line == line))]
        if bidirectional and to_station in self.adj:
            self.adj[to_station] = [e for e in self.adj[to_station] if not (e.to_station == from_station and (line is None or e.line == line))]

    def routes_from(self, station_id: str) -> List[Edge]:
        return list(self.adj.get(station_id, []))

    def stations(self) -> List[str]:
        return list(self.adj.keys())

    def is_reachable(self, src: str, dst: str) -> bool:
        if src not in self.adj or dst not in self.adj:
            return False
        visited: Set[str] = set()
        stack = [src]
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for e in self.adj.get(cur, []):
                if e.to_station not in visited:
                    stack.append(e.to_station)
        return False

    def shortest_path(self, src: str, dst: str):
        if src not in self.adj or dst not in self.adj:
            raise ValueError("Estação origem ou destino inexistente")

        pq = []
        heapq.heappush(pq, (0.0, src, None, None))
        dist: Dict[str, float] = {src: 0.0}
        prev: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        while pq:
            time_so_far, cur, prev_station, line_used = heapq.heappop(pq)
            if cur in prev:
                continue
            prev[cur] = (prev_station, line_used)
            if cur == dst:
                break
            for e in self.adj.get(cur, []):
                nxt = e.to_station
                new_time = time_so_far + e.time
                if nxt not in dist or new_time < dist[nxt]:
                    dist[nxt] = new_time
                    heapq.heappush(pq, (new_time, nxt, cur, e.line))

        if dst not in prev:
            return float('inf'), []

        path: List[Tuple[str, Optional[str]]] = []
        cur = dst
        while cur is not None:
            pst = prev[cur]
            path.append((cur, pst[1] if pst else None))
            cur = pst[0] if pst else None
        path.reverse()
        return dist[dst], path

    def fastest_route_with_transfers(self, src: str, dst: str, max_transfers: Optional[int] = None):
        if src not in self.adj or dst not in self.adj:
            raise ValueError("Estação origem ou destino inexistente")

        pq = []
        heapq.heappush(pq, (0.0, src, None, 0))
        best: Dict[Tuple[str, Optional[str], int], float] = {(src, None, 0): 0.0}
        parent: Dict[Tuple[str, Optional[str], int], Tuple[Optional[Tuple[str, Optional[str], int]], Optional[str]]] = {}

        while pq:
            time_so_far, cur, cur_line, transfers_used = heapq.heappop(pq)
            key = (cur, cur_line, transfers_used)
            if best.get(key, float('inf')) < time_so_far - 1e-9:
                continue
            if cur == dst:
                final_key = key
                final_time = time_so_far
                path_states = []
                k = final_key
                while k in parent:
                    prev_k, line_used = parent[k]
                    path_states.append((k[0], line_used))
                    k = prev_k if prev_k is not None else None
                path_states.append((src, None))
                path_states.reverse()
                return final_time, path_states

            for e in self.adj.get(cur, []):
                nxt = e.to_station
                edge_line = e.line
                travel_time = e.time
                added_wait = 0.0
                new_transfers = transfers_used
                if cur_line is None:
                    added_wait = 0.0
                else:
                    if edge_line is not None and cur_line != edge_line:
                        added_wait = self.transfer_waits.get(cur, self.default_transfer_wait)
                        new_transfers += 1
                new_time = time_so_far + added_wait + travel_time
                if max_transfers is not None and new_transfers > max_transfers:
                    continue
                new_key = (nxt, edge_line, new_transfers)
                if new_time + 1e-9 < best.get(new_key, float('inf')):
                    best[new_key] = new_time
                    parent[new_key] = (key, edge_line)
                    heapq.heappush(pq, (new_time, nxt, edge_line, new_transfers))

        return float('inf'), []

def menu():
    g = TransportGraph()
    while True:
        print("\n=== MENU TRANSPORTE ===")
        print("1. Adicionar estação")
        print("2. Remover estação")
        print("3. Adicionar conexão")
        print("4. Remover conexão")
        print("5. Consultar rotas de estação")
        print("6. Verificar se é possível chegar")
        print("7. Menor caminho (Dijkstra)")
        print("8. Melhor rota com baldeações")
        print("0. Sair")
        op = input("Escolha: ")

        if op == '1':
            nome = input("Nome da estação: ")
            g.add_station(nome)
            print("Estação adicionada!")

        elif op == '2':
            nome = input("Nome da estação: ")
            g.remove_station(nome)
            print("Estação removida!")

        elif op == '3':
            a = input("Estação origem: ")
            b = input("Estação destino: ")
            tempo = float(input("Tempo (min): "))
            linha = input("Linha (ou vazio): ") or None
            g.add_connection(a, b, tempo, linha)
            print("Conexão adicionada!")

        elif op == '4':
            a = input("Estação origem: ")
            b = input("Estação destino: ")
            linha = input("Linha (ou vazio): ") or None
            g.remove_connection(a, b, linha)
            print("Conexão removida!")

        elif op == '5':
            nome = input("Estação: ")
            for e in g.routes_from(nome):
                print(f" -> {e.to_station} ({e.time} min) via {e.line}")

        elif op == '6':
            a = input("Origem: ")
            b = input("Destino: ")
            print("Sim" if g.is_reachable(a, b) else "Não")

        elif op == '7':
            a = input("Origem: ")
            b = input("Destino: ")
            t, path = g.shortest_path(a, b)
            if t == float('inf'):
                print("Caminho não encontrado.")
            else:
                print(f"Tempo total: {t} min")
                for station, line in path:
                    print(f"  {station} (linha saída: {line})")

        elif op == '8':
            a = input("Origem: ")
            b = input("Destino: ")
            t, path = g.fastest_route_with_transfers(a, b)
            if t == float('inf'):
                print("Caminho não encontrado.")
            else:
                print(f"Tempo total: {t} min")
                for station, line in path:
                    print(f"  {station} (linha chegada: {line})")

        elif op == '0':
            break

if __name__ == "__main__":
    menu()
