#!/usr/bin/env python3
"""Deterministic generator for the benchmark's canonical datasets.

This script is the canonical source of the datasets.
No randomness is used anywhere: every value is a pure function of loop
indices, so repeated runs produce byte-identical files (LF line endings,
fixed column order). datasets/CHECKSUMS.txt pins the SHA-256 of the output.

Design constraints: every
element of the ground-truth models must be observable in the data —
  * airlines: each flight carries >=2 distinct status records (FlightStatus)
    and >=3 crew members (FlightCrew); crew members, airplanes, and
    passengers recur across flights; each location has a city column
    (Location.city); baggage belts belong to the flight's arrival airport.
  * manufacturing: operations of the same order consume different materials
    (n_material is an OrderOperation-level fact, not an Order-level one);
    quantity varies (a measure, not a constant).

Usage:
  python generate_datasets.py            # write CSVs + CHECKSUMS.txt
  python generate_datasets.py --verify   # regenerate in memory and compare
"""

import argparse
import hashlib
import io
import os
import sys
from datetime import date, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "datasets"))

AIRLINES_FILE = "airlines_ground_truth_1000.csv"
MANUF_FILE = "manufacturing_ground_truth_1000.csv"

AIRLINES_HEADER = [
    "passport", "customer_name", "customer_surname",
    "flight_number", "flight_date", "seat_number",
    "airline_code", "airline_name",
    "airport_departure", "departure_name", "departure_location", "departure_city",
    "airport_arrival", "arrival_name", "arrival_location", "arrival_city",
    "belt_number", "license_number", "airplane_model",
    "status_code", "status", "status_date", "status_hour",
    "crew_license", "crew_role",
]

MANUF_HEADER = [
    "client_id", "client_name", "n_order", "date",
    "n_material", "description_material",
    "n_operation", "description_operation",
    "workcenter_id", "order_operation_id", "timestamp", "quantity",
]

STATUSES = [
    ("SCH", "Scheduled"), ("BRD", "Boarding"), ("DEP", "Departed"),
    ("ARR", "Arrived"), ("DLY", "Delayed"),
]
SEATS = [f"{row}{col}" for row in range(1, 6) for col in "ABCDEF"]  # 30 seats

# Crew pool: 40 members with a fixed role each (FD crew_license -> role).
CREW_PILOTS = [(f"C{i:04d}", "Pilot") for i in range(1, 11)]
CREW_COPILOTS = [(f"C{i:04d}", "CoPilot") for i in range(11, 21)]
CREW_CABIN = [(f"C{i:04d}", "CabinCrew") for i in range(21, 31)] + \
             [(f"C{i:04d}", "Purser") for i in range(31, 41)]


def airlines_rows():
    """80 flights; 12-13 booking rows per flight; 1000 rows total.

    Per flight: 2-3 distinct status records and 3-5 crew members are cycled
    across the flight's rows, so multiplicity is observable within every
    flight group. Flight-level facts (airline, airports, belt, airplane)
    are constant within a group, so their functional dependencies hold.
    """
    booking = 0
    for i in range(80):
        # 20 flight numbers x 4 dates each: the same number recurs on
        # different dates, so flight_number alone is NOT a key. Dates are
        # shared by 2 flights (40 distinct dates), so flight_date alone is
        # NOT a key either -- the composite PK (flight_number, flight_date)
        # is the only minimal key the data support (accidental-key audit).
        flight_number = f"FL{i % 20 + 1:03d}"
        flight_date = (date(2026, 1, 1)
                       + timedelta(days=(i + i // 20) % 40)).isoformat()
        # Mixing in i//20 makes every flight-level fact vary across the four
        # instances of one flight number (no partial dependency on the
        # composite key, so 3NF cannot factor these onto a FlightNumber
        # entity absent from the ground truth).
        mix = i + i // 20
        # airline uses its own stride so that two same-date flights (whose mix
        # values differ by a multiple of 40) do NOT share the airline — else
        # the artifact FD flight_date -> airline_code would hold and mislead
        # every model (and the deterministic baseline) into a FlightDate entity.
        airline = (i + 3 * (i // 20)) % 10 + 1
        dep = mix % 15 + 1
        # arrival offset alternates (3 or 8, both non-zero mod 15): the same
        # departure airport reaches different destinations, so dep -/-> arr
        # (no artifact FD collapsing the two airport roles).
        arr = (mix + 3 + 5 * (i % 2)) % 15 + 1
        dep_loc = (dep - 1) % 10 + 1
        arr_loc = (arr - 1) % 10 + 1
        # Belt parity is decoupled from the arrival-offset parity (i%2), so the
        # same belt serves flights from different departure airports (no
        # artifact FD belt_number -> departure columns).
        belt = 2 * arr - ((i // 2) % 2)   # one of the arrival airport's 2 belts
        # 13 (not 1) as the replica stride: mod 5 it varies with the replica,
        # so airplane_model is NOT constant per airline (no artifact FD
        # airline_code -> airplane_model).
        plane = (i * 7 + 13 * (i // 20)) % 25 + 1  # airplanes recur across flights
        model = (plane - 1) % 5 + 1
        n_statuses = 2 + i % 2            # 2-3 distinct statuses per flight
        flight_statuses = [STATUSES[(i + t) % 5] for t in range(n_statuses)]
        n_crew = 3 + i % 3                # 1 pilot + 1 co-pilot + 1-3 cabin
        flight_crew = [CREW_PILOTS[mix % 10], CREW_COPILOTS[mix % 10]] + \
                      [CREW_CABIN[(i * 2 + t) % 20] for t in range(n_crew - 2)]
        n_bookings = 12 + i % 2           # 40 flights x12 + 40 x13 = 1000 rows
        for j in range(n_bookings):
            pax = booking * 7 % 250 + 1   # gcd(7,250)=1: all 250 passengers, 4x each
            # Same-date flights pair up (i and i+39 with the same (i+i//20)%40).
            # On the SECOND flight of each pair, the last booking re-uses the
            # first passenger of the partner flight: that passenger flies twice
            # on one date, so (passport, flight_date) is NOT an accidental key
            # of Seat (a passenger may take several flights per day).
            partner = i - 39 if i >= 39 and (i + i // 20) % 40 == ((i - 39) + (i - 39) // 20) % 40 else None
            if partner is not None and j == n_bookings - 1:
                partner_first_booking = sum(12 + k % 2 for k in range(partner))
                pax = partner_first_booking * 7 % 250 + 1
            # Names and surnames repeat across passengers (80/60 distinct), so
            # neither -- nor their pair -- is an accidental key of Customer;
            # passport -> name/surname FDs still hold (functions of pax).
            name_i = (pax - 1) % 80 + 1
            surname_i = (pax - 1) % 60 + 1
            seat = SEATS[booking * 11 % 30]
            status_code, status_name = flight_statuses[j % n_statuses]
            # Arrived/Delayed events may be recorded on the next calendar day,
            # so status_date is not column-identical to flight_date (no
            # artifact FD in either direction).
            sd = date.fromisoformat(flight_date)
            status_date = (sd + timedelta(days=1)).isoformat() \
                if status_code in ("ARR", "DLY") else flight_date
            status_hour = f"{6 + (i + (j % n_statuses) * 2) % 12:02d}:00"
            crew_license, crew_role = flight_crew[j % n_crew]
            yield [
                f"P{pax:05d}", f"Name{name_i}", f"Surname{surname_i}",
                flight_number, flight_date, seat,
                f"AL{airline:02d}", f"Airline_{airline:02d}",
                f"AP{dep:02d}", f"Airport_{dep:02d}", f"L{dep_loc:02d}", f"City_{dep_loc:02d}",
                f"AP{arr:02d}", f"Airport_{arr:02d}", f"L{arr_loc:02d}", f"City_{arr_loc:02d}",
                f"B{belt:02d}", f"REG{plane:03d}", f"Model_{model}",
                status_code, status_name, status_date, status_hour,
                crew_license, crew_role,
            ]
            booking += 1


def manufacturing_rows():
    """250 orders x 4 operation executions = 1000 rows.

    Materials vary across the operations of one order (stride 5 over 25
    materials), so material is evidently an operation-level fact; quantity
    varies; work centers rotate across orders and operations.
    """
    order_operation_id = 0
    for i in range(1, 251):
        # Client stride desynchronizes same-date order pairs (i, i+120), so
        # date -/-> client (no artifact FD); the triple-date orders 241..250
        # are overridden below to REUSE their partner's client, so
        # (client_id, date) still collides and is not an accidental key.
        client = ((i - 1) + 3 * ((i - 1) // 120)) % 40 + 1
        if i > 240:
            client = ((i - 240 - 1)) % 40 + 1
        # 120 distinct dates: orders i and i+120 share the date AND the client
        # (120 = 3x40), so neither `date` nor (client_id, date) is an
        # accidental key of Order; n_order stays the only key. The repeats
        # also make timestamp/workcenter/quantity collide across orders.
        order_date = (date(2026, 1, 1) + timedelta(days=(i - 1) % 120)).isoformat()
        # Every 5th order repeats operation type 3 instead of running type 4:
        # (n_order, n_operation) is then not unique, so the ground truth's
        # surrogate key order_operation_id is forced by the data. Some orders
        # also repeat the material / work center / quantity across their
        # operations, so no (n_order, X) pair is an accidental key either.
        # Conditions mix in i//120 so co-dated orders (i, i+120) are NOT
        # synchronized in their op/material/quantity patterns (no artifact
        # FDs of the form timestamp -> operation/quantity/...).
        op_types = [1, 2, 3, 3] if (i + i // 120) % 5 == 0 else [1, 2, 3, 4]
        mat_offsets = [0, 5, 10, 10] if (i + i // 120) % 6 == 0 else [0, 5, 10, 15]
        wc_offsets = [0, 1, 2, 2] if i % 7 == 0 else [0, 1, 2, 3]
        for j in range(1, 5):
            order_operation_id += 1
            operation = op_types[j - 1]
            material = (i + mat_offsets[j - 1] - 1) % 25 + 1
            # Orders i and i-120 share the date (hence timestamps). Giving the
            # later order's first operation the SAME material as its partner's
            # first operation makes (n_material, timestamp) collide across
            # orders, so it is not an accidental key of OrderOperation.
            if i > 120 and j == 1:
                material = (i - 120 - 1) % 25 + 1
            workcenter = (i + wc_offsets[j - 1] - 1) % 8 + 1
            timestamp = f"{order_date} 08:{15 * (j - 1):02d}:00"
            # i//120 desynchronizes quantities across co-dated orders; the
            # triple-date orders 241..250 reuse their partner's quantity
            # pattern so (quantity, timestamp) still collides somewhere.
            qi = i - 240 if i > 240 else i
            q_shift = qi + qi // 120
            quantity = 25.0 * ((q_shift + min(j, 3)) % 8 + 1) \
                if (qi + qi // 120) % 5 == 0 else 25.0 * ((q_shift + j) % 8 + 1)
            yield [
                client, f"Client_{client:02d}", i, order_date,
                material, f"Material_{material:02d}",
                operation, f"Operation_{operation:02d}",
                workcenter, order_operation_id, timestamp, quantity,
            ]


def render_csv(header, rows):
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for row in rows:
        buf.write(",".join(str(v) for v in row) + "\n")
    return buf.getvalue()


def build_all():
    return {
        AIRLINES_FILE: render_csv(AIRLINES_HEADER, airlines_rows()),
        MANUF_FILE: render_csv(MANUF_HEADER, manufacturing_rows()),
    }


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="regenerate in memory and compare with the "
                             "committed files and CHECKSUMS.txt")
    args = parser.parse_args()

    contents = build_all()
    checksums_path = os.path.join(DATASETS_DIR, "CHECKSUMS.txt")

    if args.verify:
        ok = True
        pinned = {}
        if os.path.exists(checksums_path):
            with open(checksums_path, encoding="utf-8") as f:
                for line in f:
                    digest, name = line.split()
                    pinned[name] = digest
        for name, text in contents.items():
            path = os.path.join(DATASETS_DIR, name)
            digest = sha256(text)
            with open(path, encoding="utf-8", newline="") as f:
                on_disk = f.read()
            if on_disk != text:
                print(f"MISMATCH {name}: on-disk file differs from generator output")
                ok = False
            elif pinned and pinned.get(name) != digest:
                print(f"MISMATCH {name}: CHECKSUMS.txt pin differs")
                ok = False
            else:
                print(f"OK {name} sha256={digest}")
        sys.exit(0 if ok else 1)

    lines = []
    for name, text in contents.items():
        path = os.path.join(DATASETS_DIR, name)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        digest = sha256(text)
        lines.append(f"{digest}  {name}")
        print(f"wrote {path} ({text.count(chr(10)) - 1} data rows) sha256={digest}")
    with open(checksums_path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {checksums_path}")


if __name__ == "__main__":
    main()
