from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document


CUSTOMER_SEARCH_TYPES = {
    "customer_pattern",
    "sales_strategy",
    "agent_report_decision",
    "agent_report_customer",
}


def build_customer_documents(
    prepared_log_records: list[dict[str, Any]],
    *,
    clean_text: Callable[[Any], str],
    store_name: str,
) -> list[Document]:
    documents: list[Document] = []
    for record in prepared_log_records:
        documents.append(
            Document(
                page_content=clean_text(
                    f"[상품] {clean_text(record.get('product'))} "
                    f"[고객유입패턴] {record.get('in_text') or '-'} "
                    f"[심사결과] {record.get('out_text') or '-'} "
                    f"[영업포인트] 거절사유 {record.get('reject_reason_text') or '-'} 기반으로 대안 상품 및 설득 포인트를 제안"
                )[:2000],
                metadata={
                    "type": "customer_pattern",
                    "store": store_name,
                    "product": record.get("product"),
                    "in_fields": record.get("mapped_in") or {},
                    "out_fields": record.get("mapped_out") or {},
                    "reject_reason_codes": record.get("reject_reason_codes") or [],
                    "reject_reason_details": record.get("reject_reason_details") or [],
                    "features": record.get("features") or {},
                },
            )
        )
    return documents


def format_customer_search_results(docs: list[Document]) -> list[str]:
    return [getattr(doc, "page_content", "") for doc in docs]
