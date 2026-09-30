"""Synthetic enterprise knowledge base with known relevant chunks.

Question types are written so lexical ids and paraphrases stress different retrievers.
The corpus is generated, not a dump of customer data.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from app.schemas import Chunk, Question

CORPUS_VERSION = "v1"

REGIONS = [
    ("Germany", "People Europe", "de"),
    ("Canada", "People North America", "ca"),
    ("Japan", "People Apac", "jp"),
    ("Brazil", "People Latam", "br"),
    ("Australia", "People Apac", "au"),
    ("France", "People Europe", "fr"),
    ("India", "People Apac", "in"),
    ("Mexico", "People Latam", "mx"),
    ("Singapore", "People Apac", "sg"),
    ("Ireland", "People Europe", "ie"),
    ("Sweden", "People Europe", "se"),
    ("Spain", "People Europe", "es"),
    ("Italy", "People Europe", "it"),
    ("Netherlands", "People Europe", "nl"),
    ("South Korea", "People Apac", "kr"),
    ("United Kingdom", "People Europe", "uk"),
    ("United States", "People North America", "us"),
    ("New Zealand", "People Apac", "nz"),
    ("Poland", "People Europe", "pl"),
    ("Portugal", "People Europe", "pt"),
    ("Norway", "People Europe", "no"),
    ("Denmark", "People Europe", "dk"),
    ("Finland", "People Europe", "fi"),
    ("Chile", "People Latam", "cl"),
]

PRODUCTS = [
    ("Ledger", "Treasury Platform", "ledger"),
    ("Checkout", "Commerce Experience", "checkout"),
    ("Identity", "Security Engineering", "identity"),
    ("Catalog", "Retail Platform", "catalog"),
    ("Billing", "Revenue Systems", "billing"),
    ("Notifications", "Messaging Platform", "notifications"),
    ("Search", "Discovery Platform", "search"),
    ("Inventory", "Supply Chain", "inventory"),
    ("Shipping", "Logistics", "shipping"),
    ("Rewards", "Loyalty Platform", "rewards"),
    ("Helpdesk", "Customer Operations", "helpdesk"),
    ("Analytics", "Data Platform", "analytics"),
    ("Invoicing", "Finance Systems", "invoicing"),
    ("Fraud", "Risk Engineering", "fraud"),
    ("Mobile", "Client Engineering", "mobile"),
    ("Storefront", "Web Platform", "storefront"),
    ("Subscriptions", "Revenue Systems", "subscriptions"),
    ("Tax", "Finance Systems", "tax"),
    ("Warehouse", "Logistics", "warehouse"),
    ("Recommendations", "Discovery Platform", "recommendations"),
    ("Gateway", "Security Engineering", "gateway"),
    ("Payroll", "People Systems", "payroll"),
    ("Content", "Web Platform", "content"),
    ("Returns", "Customer Operations", "returns"),
]


@dataclass(frozen=True)
class Kind:
    name: str
    code: str
    department: str
    anchors: str
    values: tuple[int, ...]
    answer: str
    extra: str
    paraphrase: str
    lexical: str
    hybrid: str
    negatives: tuple[str, str]


KINDS = [
    Kind(
        name="vacation_carryover",
        code="vac",
        department="people",
        anchors="regions",
        values=(24, 32, 40, 48, 56, 64, 72, 80),
        answer="People employed in {anchor} may carry over a maximum of {value} hours of unused vacation into the next calendar year.",
        extra="Hours above that cap are forfeited on January 1. The owning team is {team}.",
        paraphrase="For staff based in {anchor}, how much leftover annual leave can move into the following year before the surplus is wiped?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and what happens to leftover annual leave above the roll-forward limit for staff in {anchor}?",
        negatives=(
            "Staff in {anchor} are blocked from taking time away during the last week of December. {team} publishes that blackout calendar.",
            "{anchor} sick-day balances reset at the end of the medical year. {team} does not convert them into pay.",
        ),
    ),
    Kind(
        name="parental_leave",
        code="par",
        department="people",
        anchors="regions",
        values=(8, 10, 12, 14, 16, 18, 20, 26),
        answer="Employees in {anchor} are entitled to {value} weeks of parental leave paid at full base salary after a birth or adoption.",
        extra="The owning team is {team}.",
        paraphrase="How long is the fully compensated new-parent absence for people in {anchor} after a child joins the family?",
        lexical="What does policy {pid} require?",
        hybrid="Who owns policy {pid}, and how long is the fully compensated new-parent absence in {anchor}?",
        negatives=(
            "{anchor} adoption paperwork is filed with the local registry. {team} only stores the confirmation letter.",
            "New hires in {anchor} complete orientation during their first week. {team} schedules the classroom sessions.",
        ),
    ),
    Kind(
        name="expense_threshold",
        code="exp",
        department="finance",
        anchors="regions",
        values=(25, 40, 50, 75, 100, 150, 200, 250),
        answer="Expense claims filed in {anchor} above {value} dollars require itemized receipts.",
        extra="{team} rejects repayment until those documents are attached.",
        paraphrase="When must staff in {anchor} attach a detailed proof of purchase before they can be repaid?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and when must staff in {anchor} attach a detailed proof of purchase?",
        negatives=(
            "Corporate cards issued in {anchor} are billed to the local entity. {team} audits the statement monthly.",
            "Taxi rides inside {anchor} under the local cap do not need a manager signature.",
        ),
    ),
    Kind(
        name="payment_terms",
        code="net",
        department="finance",
        anchors="regions",
        values=(15, 30, 45, 60),
        answer="Vendors billing the {anchor} entity are paid on net-{value} terms.",
        extra="{team} schedules the transfer.",
        paraphrase="How many days after a supplier bill does {anchor} send payment?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and how many days after a supplier bill does {anchor} send payment?",
        negatives=(
            "{anchor} vendor onboarding requires a bank letter and a tax form. {team} stores both documents.",
            "Purchase orders for {anchor} close automatically after the goods receipt is posted.",
        ),
    ),
    Kind(
        name="badge_access",
        code="bdg",
        department="people",
        anchors="regions",
        values=(30, 45, 60, 90),
        answer="After {value} days of inactivity, entry badges for {anchor} are suspended.",
        extra="{team} performs the suspension.",
        paraphrase="When does {anchor} turn off an unused door credential?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and when does {anchor} turn off an unused door credential?",
        negatives=(
            "Visitors to the {anchor} office sign in at the front desk and receive a same-day pass.",
            "{anchor} parking permits are renewed every spring by workplace services.",
        ),
    ),
    Kind(
        name="password_rotation",
        code="pwd",
        department="security",
        anchors="products",
        values=(30, 45, 60, 90, 120),
        answer="Human accounts on {anchor} must rotate passwords every {value} days.",
        extra="{team} disables stale passwords the next morning.",
        paraphrase="How often do people have to change their sign-in secrets for {anchor}?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and how often do people change sign-in secrets for {anchor}?",
        negatives=(
            "Service accounts for {anchor} are stored in the vault and are not used by people interactively.",
            "{anchor} session cookies expire at the end of the browser day.",
        ),
    ),
    Kind(
        name="data_retention",
        code="ret",
        department="security",
        anchors="products",
        values=(14, 30, 60, 90, 180, 365),
        answer="Application logs produced by {anchor} are kept for {value} days and then destroyed.",
        extra="{team} runs the deletion job.",
        paraphrase="How long does {anchor} preserve operational event records before those records are erased?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and how long does {anchor} preserve operational event records?",
        negatives=(
            "{anchor} metrics dashboards keep graphs for a rolling quarter so on-call can compare trends.",
            "Debug traces for {anchor} are sampled and never treated as the system of record.",
        ),
    ),
    Kind(
        name="refund_window",
        code="ref",
        department="support",
        anchors="products",
        values=(14, 21, 30, 45, 60),
        answer="Customers of {anchor} may request their money back within {value} days of the original purchase.",
        extra="{team} approves the return.",
        paraphrase="What is the return window for shoppers who bought {anchor}?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and what is the return window for shoppers who bought {anchor}?",
        negatives=(
            "{anchor} shipping delays are posted on the order page. {team} does not treat a delay as a return.",
            "Gift cards bought with {anchor} cannot be exchanged at a retail counter.",
        ),
    ),
    Kind(
        name="oncall_page",
        code="page",
        department="engineering",
        anchors="products",
        values=(5, 10, 15, 20),
        answer="Alert {alert} on {anchor} means the primary replica is stale.",
        extra="Page {team} within {value} minutes and fail over to the secondary site.",
        paraphrase="How quickly must the on-call rotation take over when the {anchor} standby copy is no longer current?",
        lexical="What does alert {alert} mean?",
        hybrid="Which group owns alert {alert}, and how quickly must someone take over when the {anchor} standby copy is stale?",
        negatives=(
            "{anchor} publishes a weekly status digest. {team} writes that summary. It is not an emergency page.",
            "Capacity reviews for {anchor} happen on Thursdays and do not page the primary rotation.",
        ),
    ),
    Kind(
        name="severity_response",
        code="sev",
        department="engineering",
        anchors="products",
        values=(1, 2, 3),
        answer="Severity {value} incidents on {anchor} require a bridge call and a customer notice.",
        extra="{team} sends that notice.",
        paraphrase="Which group tells purchasers about an outage of {anchor}?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and who tells purchasers about an outage of {anchor}?",
        negatives=(
            "{anchor} maintenance windows are announced a week ahead. {team} writes those maintenance notes.",
            "Load tests for {anchor} run on the first Monday of the month and are not customer incidents.",
        ),
    ),
    Kind(
        name="deploy_freeze",
        code="frz",
        department="engineering",
        anchors="products",
        values=(5, 7, 10, 14),
        answer="Product {anchor} stops ordinary releases for {value} days before the quarter closes.",
        extra="{team} rejects non-emergency deploys during that window.",
        paraphrase="How long before a fiscal period ends does {anchor} halt routine production shipments?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and how long before a fiscal period ends does {anchor} halt routine shipments?",
        negatives=(
            "{anchor} feature flags can still be toggled by the owning engineer during normal weeks.",
            "Documentation edits for {anchor} do not go through the release train.",
        ),
    ),
    Kind(
        name="backup_rpo",
        code="rpo",
        department="engineering",
        anchors="products",
        values=(5, 15, 30, 60),
        answer="The recovery point objective for {anchor} backups is {value} minutes.",
        extra="{team} runs the restore drill each month.",
        paraphrase="What is the maximum acceptable data-loss window for {anchor} copies?",
        lexical="What does policy {pid} require?",
        hybrid="Which group owns policy {pid}, and what is the maximum acceptable data-loss window for {anchor} copies?",
        negatives=(
            "{anchor} snapshots used by developers are deleted every night and are not the disaster copy.",
            "Read replicas of {anchor} serve traffic. They are not the backup target.",
        ),
    ),
]

GENERIC = [
    "Visitors to {anchor} check in at reception and leave with a same-day paper pass.",
    "Timesheets covering {anchor} are due on Friday at 17:00 local time.",
    "{anchor} cafeteria menus change every Monday and are posted beside the elevator.",
    "Laptop refreshes for people supporting {anchor} happen on a three-year cycle.",
    "The {anchor} conference room display prefers HDMI. Adapters sit in the drawer.",
    "Floor volunteers in {anchor} organize a monthly supplies count. No policy is changed there.",
]

FAKE_PLACES = ["Atlantis", "Mars", "Olympus", "Narnia", "El Dorado", "Avalon", "Hyperion", "Zarmina"]


def _value_for(kind: Kind, slug: str) -> int:
    digest = hashlib.md5(f"{kind.name}:{slug}".encode()).hexdigest()
    return kind.values[int(digest, 16) % len(kind.values)]


def _fields(kind: Kind, anchor: str, team: str, slug: str) -> dict[str, str]:
    value = _value_for(kind, slug)
    pid = f"{kind.code}-{slug}-{value}"
    alert = f"alr-{slug}-{value}" if kind.name == "oncall_page" else ""
    return {"anchor": anchor, "team": team, "value": str(value), "pid": pid, "alert": alert}


def _gold_chunks() -> list[tuple[Chunk, dict]]:
    records = []
    groups = {"regions": REGIONS, "products": PRODUCTS}
    for kind in KINDS:
        for anchor, team, slug in groups[kind.anchors]:
            fields = _fields(kind, anchor, team, slug)
            answer = kind.answer.format(**fields)
            extra = kind.extra.format(**fields)
            chunk = Chunk(
                id=f"gold-{kind.name}-{slug}",
                text=f"Policy {fields['pid']}: {answer} {extra}",
                metadata={
                    "department": kind.department,
                    "anchor": anchor,
                    "doc_type": "policy",
                    "policy_id": fields["pid"],
                    "alert": fields["alert"],
                    "kind": kind.name,
                },
            )
            records.append((chunk, {"kind": kind, "fields": fields, "answer": f"{answer} {extra}"}))
    return records


def build_knowledge_base(n_chunks: int = 10000, seed: int = 7) -> tuple[list[Chunk], list[Question]]:
    rng = random.Random(seed)
    gold_records = _gold_chunks()
    chunks: list[Chunk] = [record[0] for record in gold_records]
    questions: list[Question] = []

    for chunk, info in gold_records:
        kind: Kind = info["kind"]
        fields = info["fields"]
        slug = chunk.id.rsplit("-", 1)[-1]
        for qtype, template in (
            ("paraphrase", kind.paraphrase),
            ("lexical", kind.lexical),
            ("hybrid", kind.hybrid),
        ):
            questions.append(
                Question(
                    id=f"{qtype}-{kind.name}-{slug}",
                    text=template.format(**fields),
                    gold_ids=[chunk.id],
                    answer=info["answer"],
                    qtype=qtype,
                    department=kind.department,
                )
            )
        for index, template in enumerate(kind.negatives):
            chunks.append(
                Chunk(
                    id=f"hard-{kind.name}-{slug}-{index}",
                    text=template.format(**fields),
                    metadata={
                        "department": kind.department,
                        "anchor": fields["anchor"],
                        "doc_type": "hard_negative",
                        "policy_id": "",
                        "alert": "",
                        "kind": kind.name,
                    },
                )
            )

    by_anchor_group = {"regions": [], "products": []}
    for kind in KINDS:
        by_anchor_group[kind.anchors].append(kind)
    groups = {"regions": REGIONS, "products": PRODUCTS}
    for group_name, kinds in by_anchor_group.items():
        pairs = [(kinds[i], kinds[i + 1]) for i in range(0, len(kinds) - 1, 2)]
        if len(kinds) % 2 == 1:
            pairs.append((kinds[-2], kinds[-1]))
        for anchor, team, slug in groups[group_name]:
            for left, right in pairs:
                left_fields = _fields(left, anchor, team, slug)
                right_fields = _fields(right, anchor, team, slug)
                left_id = f"gold-{left.name}-{slug}"
                right_id = f"gold-{right.name}-{slug}"
                left_answer = f"{left.answer.format(**left_fields)} {left.extra.format(**left_fields)}"
                right_answer = f"{right.answer.format(**right_fields)} {right.extra.format(**right_fields)}"
                lexical = right.lexical.format(**right_fields)
                lexical = lexical[0].lower() + lexical[1:]
                department = left.department if left.department == right.department else ""
                questions.append(
                    Question(
                        id=f"compound-{left.name}-{right.name}-{slug}",
                        text=f"{left.paraphrase.format(**left_fields)} Additionally, {lexical}",
                        gold_ids=[left_id, right_id],
                        answer=f"{left_answer} {right_answer}",
                        qtype="compound",
                        department=department,
                    )
                )

    for anchor, _team, slug in REGIONS + PRODUCTS:
        department = "people" if any(anchor == name for name, _t, _s in REGIONS) else "engineering"
        for index, template in enumerate(GENERIC):
            chunks.append(
                Chunk(
                    id=f"gen-{slug}-{index}",
                    text=template.format(anchor=anchor),
                    metadata={
                        "department": department,
                        "anchor": anchor,
                        "doc_type": "distractor",
                        "policy_id": "",
                        "alert": "",
                        "kind": "generic",
                    },
                )
            )

    filler_index = 0
    while len(chunks) < n_chunks:
        chunks.append(
            Chunk(
                id=f"fill-{filler_index}",
                text=(
                    f"Meeting note MN-{filler_index}: attendees reviewed cafeteria menu item {filler_index} "
                    f"and the printer on floor {filler_index % 7}. No decision was recorded. Token NOTE-{filler_index}."
                ),
                metadata={
                    "department": "operations",
                    "anchor": "",
                    "doc_type": "note",
                    "policy_id": "",
                    "alert": "",
                    "kind": "filler",
                },
            )
        )
        filler_index += 1

    if len(chunks) > n_chunks:
        gold = [chunk for chunk in chunks if chunk.metadata["doc_type"] == "policy"]
        rest = [chunk for chunk in chunks if chunk.metadata["doc_type"] != "policy"]
        rng.shuffle(rest)
        keep = n_chunks - len(gold)
        if keep < 0:
            rng.shuffle(gold)
            chunks = gold[:n_chunks]
        else:
            chunks = gold + rest[:keep]

    gold_ids = {chunk.id for chunk in chunks}
    questions = [question for question in questions if all(gid in gold_ids for gid in question.gold_ids)]

    for index, place in enumerate(FAKE_PLACES):
        questions.append(
            Question(
                id=f"unanswerable-{index}",
                text=f"What is the lunar mining allowance for staff based in {place}?",
                gold_ids=[],
                answer="",
                qtype="unanswerable",
                department="",
            )
        )
        questions.append(
            Question(
                id=f"unanswerable-policy-{index}",
                text=f"What does policy zzz-{place.lower()}-999 require?",
                gold_ids=[],
                answer="",
                qtype="unanswerable",
                department="",
            )
        )

    rng.shuffle(chunks)
    return chunks, questions


def split_questions(questions: list[Question], seed: int = 7) -> dict[str, list[Question]]:
    """Hold out a development slice for the abstention threshold. Test questions are disjoint."""
    rng = random.Random(seed)
    buckets: dict[str, list[Question]] = {}
    for question in questions:
        buckets.setdefault(question.qtype, []).append(question)
    for pool in buckets.values():
        rng.shuffle(pool)

    def take(qtype: str, count: int) -> list[Question]:
        pool = buckets[qtype]
        if len(pool) < count:
            raise RuntimeError(f"need {count} {qtype} questions, found {len(pool)}")
        chosen = pool[:count]
        del pool[:count]
        return chosen

    dev = take("paraphrase", 30) + take("lexical", 30) + take("hybrid", 30) + take("unanswerable", 20)
    test = (
        take("paraphrase", 80)
        + take("lexical", 80)
        + take("hybrid", 80)
        + take("compound", 60)
        + take("unanswerable", 40)
    )
    for index, question in enumerate(test):
        if question.qtype in {"paraphrase", "lexical", "hybrid"} and question.department and index % 5 == 0:
            question.filters = {"department": question.department}
    rng.shuffle(dev)
    rng.shuffle(test)
    return {"dev": dev, "test": test}
