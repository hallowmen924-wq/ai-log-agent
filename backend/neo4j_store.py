from __future__ import annotations

import os
from typing import Any


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


class Neo4jGraphStore:
    """Optional Neo4j graph adapter.

    - Disabled by default unless USE_NEO4J_GRAPH=1 and connection envs are present.
    - All public methods are fail-safe and return empty data on error.
    """

    def __init__(self) -> None:
        self.enabled = _env_flag("USE_NEO4J_GRAPH", "0")
        self.uri = str(os.getenv("NEO4J_URI", "")).strip()
        self.username = str(os.getenv("NEO4J_USERNAME", "")).strip()
        self.password = str(os.getenv("NEO4J_PASSWORD", "")).strip()
        self.database = str(os.getenv("NEO4J_DATABASE", "neo4j")).strip() or "neo4j"
        self._driver = None
        self._init_error = ""
        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        if not (self.uri and self.username and self.password):
            self._init_error = "missing neo4j connection envs"
            self.enabled = False
            return
        try:
            from neo4j import GraphDatabase  # type: ignore

            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=20,
                connection_timeout=8,
            )
            self._driver.verify_connectivity()
        except Exception as exc:  # pragma: no cover
            self._init_error = str(exc)
            self.enabled = False
            self._driver = None

    @property
    def available(self) -> bool:
        return bool(self.enabled and self._driver is not None)

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "uri": self.uri,
            "database": self.database,
            "error": self._init_error,
        }

    def close(self) -> None:
        try:
            if self._driver is not None:
                self._driver.close()
        except Exception:
            pass

    def rebuild(self, records: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.available:
            return {"ok": False, "reason": "neo4j unavailable"}
        try:
            feature_rows = [
                {
                    "feature_id": str(item.get("feature_id") or ""),
                    "feature_name": str(item.get("feature_name") or item.get("feature_id") or ""),
                    "category": str(item.get("category") or ""),
                }
                for item in (features or [])
                if str(item.get("feature_id") or "").strip()
            ]
            record_rows = []
            feature_links = []
            for item in (records or []):
                record_id = str(item.get("record_id") or "").strip()
                if not record_id:
                    continue
                row = {
                    "record_id": record_id,
                    "product": str(item.get("product") or ""),
                    "decision": str(item.get("decision") or ""),
                    "age_band": str(item.get("age_band") or ""),
                    "income_band": str(item.get("income_band") or ""),
                    "amount_band": str(item.get("amount_band") or ""),
                    "rate": item.get("rate"),
                    "amount": item.get("amount"),
                    "search_text": str(item.get("search_text") or ""),
                }
                record_rows.append(row)
                for fid in item.get("feature_ids") or []:
                    fid_norm = str(fid or "").strip()
                    if fid_norm:
                        feature_links.append({"record_id": record_id, "feature_id": fid_norm})

            with self._driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")

                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (f:Feature {feature_id: row.feature_id})
                    SET f.feature_name = row.feature_name,
                        f.category = row.category
                    """,
                    rows=feature_rows,
                )
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (r:Record {record_id: row.record_id})
                    SET r.product = row.product,
                        r.decision = row.decision,
                        r.age_band = row.age_band,
                        r.income_band = row.income_band,
                        r.amount_band = row.amount_band,
                        r.rate = row.rate,
                        r.amount = row.amount,
                        r.search_text = row.search_text
                    """,
                    rows=record_rows,
                )
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (r:Record {record_id: row.record_id})
                    MATCH (f:Feature {feature_id: row.feature_id})
                    MERGE (r)-[:HAS_FEATURE]->(f)
                    """,
                    rows=feature_links,
                )
                session.run("CREATE INDEX record_product IF NOT EXISTS FOR (r:Record) ON (r.product)")
                session.run("CREATE INDEX feature_id IF NOT EXISTS FOR (f:Feature) ON (f.feature_id)")

            return {
                "ok": True,
                "features": len(feature_rows),
                "records": len(record_rows),
                "feature_links": len(feature_links),
            }
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "reason": str(exc)}

    def query_customer_clusters(self, *, selected_product: str, decision_focus: str = "", limit: int = 20) -> list[dict[str, Any]]:
        if not self.available:
            return []
        cypher = """
        MATCH (r:Record)
        WHERE ($product = '' OR r.product = $product)
          AND ($decision = '' OR r.decision = $decision)
        WITH r.product AS product, r.decision AS decision, r.age_band AS age_band, r.income_band AS income_band, r.amount_band AS amount_band,
             count(*) AS cnt, avg(coalesce(toFloat(r.rate), 0.0)) AS avg_rate, avg(coalesce(toFloat(r.amount), 0.0)) AS avg_amount
        RETURN product, decision, age_band, income_band, amount_band, cnt, avg_rate, avg_amount
        ORDER BY cnt DESC
        LIMIT $limit
        """
        try:
            rows: list[dict[str, Any]] = []
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    cypher,
                    product=selected_product or "",
                    decision=decision_focus or "",
                    limit=max(1, int(limit)),
                )
                for rec in result:
                    rows.append(
                        {
                            "product": rec.get("product") or "",
                            "decision": rec.get("decision") or "",
                            "age_band": rec.get("age_band") or "",
                            "income_band": rec.get("income_band") or "",
                            "amount_band": rec.get("amount_band") or "",
                            "count": int(rec.get("cnt") or 0),
                            "avg_rate": float(rec.get("avg_rate") or 0.0),
                            "avg_amount": float(rec.get("avg_amount") or 0.0),
                        }
                    )
            return rows
        except Exception:
            return []

    def query_records_by_feature(
        self,
        *,
        selected_product: str,
        feature_ids: list[str],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        if not self.available or not feature_ids:
            return []
        cypher = """
        MATCH (r:Record)-[:HAS_FEATURE]->(f:Feature)
        WHERE ($product = '' OR r.product = $product)
          AND f.feature_id IN $feature_ids
        WITH r, count(DISTINCT f.feature_id) AS match_count
        RETURN r.record_id AS record_id, r.product AS product, r.decision AS decision, r.rate AS rate, r.amount AS amount, r.search_text AS search_text, match_count
        ORDER BY match_count DESC, record_id ASC
        LIMIT $limit
        """
        try:
            rows: list[dict[str, Any]] = []
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    cypher,
                    product=selected_product or "",
                    feature_ids=[str(item) for item in feature_ids if str(item).strip()],
                    limit=max(1, int(limit)),
                )
                for rec in result:
                    rows.append(
                        {
                            "record_id": rec.get("record_id"),
                            "product": rec.get("product"),
                            "decision": rec.get("decision"),
                            "score": int(rec.get("match_count") or 0),
                            "rate": rec.get("rate"),
                            "amount": rec.get("amount"),
                            "snippet": str(rec.get("search_text") or "")[:260],
                            "reject_codes": [],
                            "reject_descriptions": [],
                        }
                    )
            return rows
        except Exception:
            return []
