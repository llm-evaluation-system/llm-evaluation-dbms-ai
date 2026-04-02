#!/usr/bin/env python3
"""
scripts/init_test_db.py — Initialise the sandboxed test PostgreSQL with all
schemas and dummy data required by the question bank SQL fixtures.

Run this ONCE after `docker compose up -d` and before any evaluation:

    # Inside the app container (recommended):
    docker compose exec app python scripts/init_test_db.py

    # Or from your host machine (connects via host-mapped port 5433):
    python scripts/init_test_db.py

Design:
  - All CREATE TABLE statements use IF NOT EXISTS — fully idempotent.
  - All INSERT statements use ON CONFLICT DO NOTHING — safe to re-run.
  - Tables that could collide across questions use question-ID suffixes
    (e.g. rel_a_7f30, rel_r_7f30) so they coexist without conflict.
  - The script reports which schemas were applied and any errors encountered.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg

# Prefer the OS environment variable (set correctly by docker-compose to use
# the service name 'db_test:5432') over the value in .env (which uses
# 'localhost:5433' for host-machine access).  This makes the script work
# correctly whether run inside the container or from the host.
from config import TEST_DATABASE_URL
_TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", TEST_DATABASE_URL)


# ---------------------------------------------------------------------------
# All DDL blocks to apply to the test DB — ORDER MATTERS for FK dependencies
# ---------------------------------------------------------------------------

INIT_BLOCKS: list[tuple[str, str]] = [
    # ── (label, DDL) ──────────────────────────────────────────────────────

    ("students", """
CREATE TABLE IF NOT EXISTS students (
    sid     INTEGER PRIMARY KEY,
    sname   VARCHAR(64),
    login   VARCHAR(64),
    age     INTEGER,
    gpa     REAL
);
INSERT INTO students VALUES
  (50000, 'Dave',    'dave@cs',         19, 3.3),
  (53666, 'Jones',   'jones@cs',        18, 3.4),
  (53688, 'Smith',   'smith@ee',        18, 3.2),
  (53650, 'Smith',   'smith@math',      19, 3.8),
  (53831, 'Madayan', 'madayan@music',   11, 1.8),
  (53832, 'Guldu',   'guldu@music',     12, 2.0)
ON CONFLICT DO NOTHING;
"""),

    ("sailors", """
CREATE TABLE IF NOT EXISTS sailors (
    sid     INTEGER PRIMARY KEY,
    sname   VARCHAR(32),
    rating  INTEGER,
    age     REAL
);
INSERT INTO sailors VALUES
  (18, 'jones', 3,    30.0),
  (41, 'jonah', 6,    56.0),
  (22, 'ahab',  7,    44.0),
  (63, 'moby',  NULL, 15.0)
ON CONFLICT DO NOTHING;
"""),

    ("reserves", """
CREATE TABLE IF NOT EXISTS reserves (
    sid  INTEGER,
    bid  INTEGER,
    day  DATE,
    PRIMARY KEY (sid, bid, day)
);
INSERT INTO reserves VALUES
  (22, 101, '1996-10-10'),
  (22, 102, '1996-10-10'),
  (22, 103, '1996-08-08'),
  (22, 104, '1996-07-07'),
  (31, 102, '1996-10-11'),
  (31, 103, '1996-06-06'),
  (31, 104, '1996-08-08'),
  (64, 101, '1996-09-05'),
  (64, 102, '1996-09-08'),
  (74, 103, '1996-09-08')
ON CONFLICT DO NOTHING;
"""),

    ("emp (shared)", """
CREATE TABLE IF NOT EXISTS emp (
    eid       INTEGER PRIMARY KEY,
    ename     VARCHAR(64),
    age       INTEGER,
    salary    REAL
);
INSERT INTO emp VALUES
  (1, 'Alice',   45, 120000),
  (2, 'Bob',     32, 85000),
  (3, 'Charlie', 55, 150000),
  (4, 'Diana',   28, 60000),
  (5, 'Eve',     41, 95000),
  (6, 'Frank',   35, 75000)
ON CONFLICT DO NOTHING;
"""),

    ("dept (shared)", """
CREATE TABLE IF NOT EXISTS dept (
    did       INTEGER PRIMARY KEY,
    budget    REAL,
    managerid INTEGER REFERENCES emp(eid)
);
INSERT INTO dept VALUES
  (10, 500000, 1),
  (20, 300000, 3),
  (30, 200000, 5)
ON CONFLICT DO NOTHING;
"""),

    ("works", """
CREATE TABLE IF NOT EXISTS works (
    eid      INTEGER REFERENCES emp(eid),
    did      INTEGER REFERENCES dept(did),
    pct_time INTEGER,
    PRIMARY KEY (eid, did)
);
INSERT INTO works VALUES
  (1, 10, 100), (2, 10, 50), (2, 20, 50),
  (3, 20, 100), (4, 30, 100), (5, 30, 80), (6, 10, 100)
ON CONFLICT DO NOTHING;
"""),

    ("senioremp view", """
CREATE OR REPLACE VIEW senioremp AS
    SELECT e.ename AS sname, e.age AS sage, e.salary
    FROM emp e
    WHERE e.age > 50;
"""),

    ("suppliers", """
CREATE TABLE IF NOT EXISTS suppliers (
    sid     INTEGER PRIMARY KEY,
    sname   VARCHAR(64),
    address VARCHAR(128)
);
INSERT INTO suppliers VALUES
  (1, 'Yosemite Sham',   '221 Packer Ave'),
  (2, 'BigBolt Inc',     '500 Main St'),
  (3, 'FastParts Ltd',   '99 Industrial Blvd'),
  (4, 'GreenParts Co',   '10 Green St')
ON CONFLICT DO NOTHING;
"""),

    ("parts", """
CREATE TABLE IF NOT EXISTS parts (
    pid   INTEGER PRIMARY KEY,
    pname VARCHAR(64),
    color VARCHAR(32)
);
INSERT INTO parts VALUES
  (1, 'NutA',   'red'),
  (2, 'BoltB',  'green'),
  (3, 'WasherC','red'),
  (4, 'ScrewD', 'blue'),
  (5, 'NutE',   'green'),
  (6, 'BoltF',  'red')
ON CONFLICT DO NOTHING;
"""),

    ("catalog", """
CREATE TABLE IF NOT EXISTS catalog (
    sid  INTEGER REFERENCES suppliers(sid),
    pid  INTEGER REFERENCES parts(pid),
    cost REAL,
    PRIMARY KEY (sid, pid)
);
INSERT INTO catalog VALUES
  (1,1,10),(1,2,20),(1,3,15),(2,1,8),(2,4,12),
  (3,2,18),(3,5,22),(4,3,14),(4,6,30),(1,5,25),(2,6,28)
ON CONFLICT DO NOTHING;
"""),

    ("aircraft", """
CREATE TABLE IF NOT EXISTS aircraft (
    aid           INTEGER PRIMARY KEY,
    aname         VARCHAR(64),
    cruisingrange INTEGER
);
INSERT INTO aircraft VALUES
  (1,'Boeing 747',8000),(2,'Boeing 737',4000),
  (3,'Airbus A320',5500),(4,'Cessna 172',800),(5,'Boeing 777',10000)
ON CONFLICT DO NOTHING;
"""),

    ("femployees (flight employees)", """
CREATE TABLE IF NOT EXISTS femployees (
    eid    INTEGER PRIMARY KEY,
    ename  VARCHAR(64),
    salary INTEGER
);
INSERT INTO femployees VALUES
  (1,'PilotAl',120000),(2,'PilotBob',95000),
  (3,'PilotCal',75000),(4,'EngDave',60000),(5,'PilotEve',110000)
ON CONFLICT DO NOTHING;
"""),

    ("certified", """
CREATE TABLE IF NOT EXISTS certified (
    eid INTEGER REFERENCES femployees(eid),
    aid INTEGER REFERENCES aircraft(aid),
    PRIMARY KEY (eid, aid)
);
INSERT INTO certified VALUES
  (1,1),(1,2),(1,5),(2,2),(2,3),(3,3),(3,4),(5,1),(5,5)
ON CONFLICT DO NOTHING;
"""),

    ("flights", """
CREATE TABLE IF NOT EXISTS flights (
    flno     INTEGER PRIMARY KEY,
    ffrom    VARCHAR(64),
    fto      VARCHAR(64),
    distance INTEGER,
    departs  TIME,
    arrives  TIME,
    price    INTEGER
);
INSERT INTO flights VALUES
  (1,'Madison','Chicago',300,'08:00','09:30',250),
  (2,'Chicago','New York',800,'10:00','13:00',450),
  (3,'Madison','New York',900,'07:00','11:00',500),
  (4,'Bonn','Madras',7500,'12:00','23:00',1200),
  (5,'Madison','Los Angeles',2000,'06:00','09:00',700),
  (6,'Los Angeles','Honolulu',2500,'10:00','14:00',800),
  (7,'New York','Madison',900,'15:00','19:00',500),
  (8,'Chicago','Los Angeles',2200,'11:00','14:30',650)
ON CONFLICT DO NOTHING;
"""),

    ("student", """
CREATE TABLE IF NOT EXISTS student (
    snum  INTEGER PRIMARY KEY,
    sname VARCHAR(64),
    major VARCHAR(64),
    level VARCHAR(8),
    age   INTEGER
);
INSERT INTO student VALUES
  (1001,'Alice','CS','JR',20),(1002,'Bob','Math','SR',22),
  (1003,'Carol','CS','JR',21),(1004,'Dave','History','SO',19),
  (1005,'Eve','CS','FR',18),(1006,'Frank','Physics','JR',20),
  (1007,'Grace','Math','SR',23)
ON CONFLICT DO NOTHING;
"""),

    ("faculty", """
CREATE TABLE IF NOT EXISTS faculty (
    fid    INTEGER PRIMARY KEY,
    fname  VARCHAR(64),
    deptid INTEGER
);
INSERT INTO faculty VALUES
  (10,'I.Teach',1),(20,'Prof.Oak',2),(30,'Dr.Lake',1)
ON CONFLICT DO NOTHING;
"""),

    ("class", """
CREATE TABLE IF NOT EXISTS class (
    name     VARCHAR(64) PRIMARY KEY,
    meets_at TIME,
    room     VARCHAR(16),
    fid      INTEGER REFERENCES faculty(fid)
);
INSERT INTO class VALUES
  ('Databases','09:00','R128',10),('Algorithms','10:00','R200',20),
  ('OS','11:00','R128',10),('Calculus','13:00','R300',30),
  ('LinearAlg','14:00','R128',30)
ON CONFLICT DO NOTHING;
"""),

    ("enrolled", """
CREATE TABLE IF NOT EXISTS enrolled (
    snum  INTEGER REFERENCES student(snum),
    cname VARCHAR(64) REFERENCES class(name),
    PRIMARY KEY (snum, cname)
);
INSERT INTO enrolled VALUES
  (1001,'Databases'),(1001,'Algorithms'),(1002,'Databases'),
  (1002,'Calculus'),(1003,'Databases'),(1003,'OS'),
  (1004,'Calculus'),(1004,'LinearAlg'),(1005,'Databases'),
  (1005,'OS'),(1006,'Algorithms'),(1006,'Calculus'),
  (1007,'Databases'),(1007,'LinearAlg')
ON CONFLICT DO NOTHING;
"""),

    ("emp_qe (query evaluation Emp)", """
CREATE TABLE IF NOT EXISTS emp_qe (
    eid   INTEGER PRIMARY KEY,
    did   INTEGER,
    sal   INTEGER,
    hobby VARCHAR(32)
);
INSERT INTO emp_qe VALUES
  (1,10,55000,'tennis'),(2,10,62000,'yodeling'),(3,20,48000,'chess'),
  (4,20,71000,'tennis'),(5,30,58000,'yodeling'),(6,30,44000,'chess'),
  (7,10,39000,'golf'),(8,20,83000,'yodeling'),
  (9,30,95000,'tennis'),(10,10,27000,'chess')
ON CONFLICT DO NOTHING;
"""),

    ("dept_qe (query evaluation Dept)", """
CREATE TABLE IF NOT EXISTS dept_qe (
    did    INTEGER PRIMARY KEY,
    dname  VARCHAR(32),
    floor  INTEGER,
    budget REAL
);
INSERT INTO dept_qe VALUES
  (10,'Engineering',1,400000),(20,'Marketing',2,250000),(30,'Finance',1,300000)
ON CONFLICT DO NOTHING;
"""),

    ("emp_idx (indexing Emp)", """
CREATE TABLE IF NOT EXISTS emp_idx (
    eid     INTEGER PRIMARY KEY,
    ename   VARCHAR(64),
    sal     INTEGER,
    age     INTEGER,
    did     INTEGER
);
INSERT INTO emp_idx VALUES
  (1,'Alice',90000,45,10),(2,'Bob',55000,32,10),
  (3,'Charlie',78000,55,20),(4,'Diana',42000,28,20),
  (5,'Eve',110000,41,30),(6,'Frank',63000,35,30),
  (7,'Grace',88000,50,10),(8,'Hank',15000,25,20)
ON CONFLICT DO NOTHING;
"""),

    ("dept_idx (indexing Dept)", """
CREATE TABLE IF NOT EXISTS dept_idx (
    did     INTEGER PRIMARY KEY,
    budget  INTEGER,
    floor   INTEGER,
    mgr_eid INTEGER
);
INSERT INTO dept_idx VALUES
  (10,500000,1,1),(20,300000,3,3),(30,200000,10,5)
ON CONFLICT DO NOTHING;
"""),

    ("executives", """
CREATE TABLE IF NOT EXISTS executives (
    ename   VARCHAR(64),
    title   VARCHAR(64),
    dname   VARCHAR(64),
    address VARCHAR(128)
);
INSERT INTO executives VALUES
  ('Alice Smith','CEO','Engineering','100 Main St'),
  ('Bob Jones','CTO','Engineering','200 Elm St'),
  ('Carol White','CFO','Finance','300 Oak Ave'),
  ('Dave Brown','COO','Operations','400 Pine Rd'),
  ('Eve Davis','VP','Marketing','500 Maple Dr'),
  ('Frank Miller','VP','Engineering','600 Cedar Ln')
ON CONFLICT DO NOTHING;
"""),

    ("finance", """
CREATE TABLE IF NOT EXISTS finance (
    did      INTEGER PRIMARY KEY,
    budget   REAL,
    sales    REAL,
    expenses REAL
);
INSERT INTO finance VALUES
  (10,400000,900000,500000),(20,250000,600000,350000),(30,300000,750000,400000)
ON CONFLICT DO NOTHING;
"""),

    ("parts_design (DB design)", """
CREATE TABLE IF NOT EXISTS parts_design (
    pid       INTEGER PRIMARY KEY,
    pname     VARCHAR(64),
    cost      REAL,
    num_avail INTEGER
);
INSERT INTO parts_design VALUES
  (1,'nuts',2.50,1000),(2,'bolts',1.80,500),(3,'washer',0.50,2000),
  (4,'screw',0.30,3000),(5,'nail',0.10,5000)
ON CONFLICT DO NOTHING;
"""),

    ("professors", """
CREATE TABLE IF NOT EXISTS professors (
    prof_ssn   CHAR(10) PRIMARY KEY,
    name       VARCHAR(64),
    age        INTEGER,
    rank       INTEGER,
    speciality VARCHAR(64)
);
INSERT INTO professors VALUES
  ('111-22-333','Prof. Adams',52,3,'Databases'),
  ('222-33-444','Prof. Baker',45,2,'AI'),
  ('333-44-555','Prof. Clark',38,1,'Networks')
ON CONFLICT DO NOTHING;
"""),

    ("depts (university)", """
CREATE TABLE IF NOT EXISTS depts (
    dno    INTEGER PRIMARY KEY,
    dname  VARCHAR(64),
    office VARCHAR(16)
);
INSERT INTO depts VALUES
  (1,'Computer Science','A101'),(2,'Mathematics','B202'),(3,'Physics','C303')
ON CONFLICT DO NOTHING;
"""),

    ("projects", """
CREATE TABLE IF NOT EXISTS projects (
    pid        INTEGER PRIMARY KEY,
    sponsor    VARCHAR(64),
    start_date DATE,
    end_date   DATE,
    budget     REAL
);
INSERT INTO projects VALUES
  (101,'NSF','2022-01-01','2024-12-31',500000),
  (102,'DARPA','2021-06-01','2023-05-31',1200000),
  (103,'NIH','2023-03-01','2025-02-28',750000)
ON CONFLICT DO NOTHING;
"""),

    ("graduates", """
CREATE TABLE IF NOT EXISTS graduates (
    grad_ssn VARCHAR(10) PRIMARY KEY,
    name     VARCHAR(64),
    age      INTEGER,
    deg_prog VARCHAR(32)
);
INSERT INTO graduates VALUES
  ('444-55-666','Grad Ann',26,'PhD'),
  ('555-66-777','Grad Bill',24,'MS'),
  ('666-77-888','Grad Cara',27,'PhD')
ON CONFLICT DO NOTHING;
"""),

    ("emp_bcnf (DB design/tuning)", """
CREATE TABLE IF NOT EXISTS emp_bcnf (
    eid    INTEGER PRIMARY KEY,
    ename  VARCHAR(64),
    addr   VARCHAR(128),
    sal    REAL,
    age    INTEGER,
    yrs    INTEGER,
    deptid INTEGER
);
INSERT INTO emp_bcnf VALUES
  (1,'Alice','1 Main St',120000,45,20,10),
  (2,'Bob','2 Elm St',85000,32,8,10),
  (3,'Charlie','3 Oak Ave',150000,55,30,20),
  (4,'Diana','4 Pine Rd',60000,28,3,20),
  (5,'Eve','5 Maple Dr',95000,41,15,30),
  (6,'Frank','6 Cedar Ln',75000,35,10,30),
  (7,'Grace','7 Birch Ct',110000,50,22,10),
  (8,'Hank','8 Spruce Pl',40000,26,2,20)
ON CONFLICT DO NOTHING;
"""),

    ("dept_bcnf (DB design/tuning)", """
CREATE TABLE IF NOT EXISTS dept_bcnf (
    did    INTEGER PRIMARY KEY,
    dname  VARCHAR(64),
    floor  INTEGER,
    budget REAL
);
INSERT INTO dept_bcnf VALUES
  (10,'Engineering',1,500000),(20,'Marketing',2,300000),(30,'Finance',3,200000)
ON CONFLICT DO NOTHING;
"""),

    ("mgrAge view", """
CREATE OR REPLACE VIEW mgrAge AS
    SELECT d.dname, e.age
    FROM emp_bcnf e
    JOIN dept_bcnf d ON d.did = e.deptid;
"""),

    ("dept_mgr (security/manager)", """
CREATE TABLE IF NOT EXISTS dept_mgr (
    did       INTEGER PRIMARY KEY,
    dname     VARCHAR(64),
    location  VARCHAR(64),
    managerid INTEGER
);
INSERT INTO dept_mgr VALUES
  (10,'Engineering','NYC',1),(20,'Marketing','NYC',3),
  (30,'Finance','SF',5),(40,'Operations','SF',2),(50,'R&D','Boston',4)
ON CONFLICT DO NOTHING;
"""),

    ("emp_mgr (security/manager)", """
CREATE TABLE IF NOT EXISTS emp_mgr (
    eid INTEGER PRIMARY KEY,
    sal REAL
);
INSERT INTO emp_mgr VALUES
  (1,120000),(2,85000),(3,150000),(4,60000),(5,95000)
ON CONFLICT DO NOTHING;
"""),

    ("times_dim (OLAP)", """
CREATE TABLE IF NOT EXISTS times_dim (
    timeid INTEGER PRIMARY KEY,
    year   INTEGER,
    month  INTEGER,
    day    INTEGER
);
INSERT INTO times_dim VALUES
  (1,2020,1,15),(2,2020,6,20),(3,2021,3,10),(4,2021,9,25),(5,2022,2,28)
ON CONFLICT DO NOTHING;
"""),

    ("sales_fact (OLAP)", """
CREATE TABLE IF NOT EXISTS sales_fact (
    saleid INTEGER PRIMARY KEY,
    timeid INTEGER REFERENCES times_dim(timeid),
    sales  REAL
);
INSERT INTO sales_fact VALUES
  (1,1,100000),(2,2,150000),(3,3,120000),(4,4,180000),(5,5,200000)
ON CONFLICT DO NOTHING;
"""),

    ("numreservations view", """
CREATE OR REPLACE VIEW numreservations AS
    SELECT s.sid, s.sname, COUNT(*) AS numres
    FROM sailors s
    JOIN reserves r ON s.sid = r.sid
    GROUP BY s.sid, s.sname;
"""),

    # ── Question-specific tables (suffixed to avoid collisions) ──────────────

    ("rel_a_7f30 (ternary relationship A)", """
CREATE TABLE IF NOT EXISTS rel_a_7f30 (
    a1 CHAR(10) PRIMARY KEY,
    a2 CHAR(10)
);
INSERT INTO rel_a_7f30 VALUES ('a001','ax1'),('a002','ax2') ON CONFLICT DO NOTHING;
"""),

    ("rel_b_7f30 (ternary relationship B)", """
CREATE TABLE IF NOT EXISTS rel_b_7f30 (
    b1 CHAR(10) PRIMARY KEY,
    b2 CHAR(10)
);
INSERT INTO rel_b_7f30 VALUES ('b001','bx1'),('b002','bx2') ON CONFLICT DO NOTHING;
"""),

    ("rel_c_7f30 (ternary relationship C)", """
CREATE TABLE IF NOT EXISTS rel_c_7f30 (
    c1 CHAR(10) PRIMARY KEY,
    c2 CHAR(10)
);
INSERT INTO rel_c_7f30 VALUES ('c001','cx1'),('c002','cx2') ON CONFLICT DO NOTHING;
"""),

    ("rel_r_7f30 (ternary relationship R merged into A)", """
CREATE TABLE IF NOT EXISTS rel_r_7f30 (
    a1 CHAR(10) PRIMARY KEY,
    b1 CHAR(10) UNIQUE,
    c1 CHAR(10),
    FOREIGN KEY (a1) REFERENCES rel_a_7f30(a1),
    FOREIGN KEY (b1) REFERENCES rel_b_7f30(b1),
    FOREIGN KEY (c1) REFERENCES rel_c_7f30(c1)
);
INSERT INTO rel_r_7f30 VALUES ('a001','b001','c001') ON CONFLICT DO NOTHING;
"""),

    ("rel_r_5893 (projection/join analysis R)", """
CREATE TABLE IF NOT EXISTS rel_r_5893 (
    a     INTEGER PRIMARY KEY,
    b     INTEGER,
    c_attr INTEGER,
    d_attr INTEGER
);
INSERT INTO rel_r_5893 VALUES
  (1,10,1,100),(2,20,2,200),(3,30,3,300),(4,40,1,400),(5,50,2,500)
ON CONFLICT DO NOTHING;
"""),

    ("rel_s_5893 (projection/join analysis S)", """
CREATE TABLE IF NOT EXISTS rel_s_5893 (
    c_attr INTEGER PRIMARY KEY,
    d_attr INTEGER,
    e_attr INTEGER,
    f_attr INTEGER,
    g_attr INTEGER
);
INSERT INTO rel_s_5893 VALUES
  (1,100,1000,10000,100000),(2,200,2000,20000,200000),(3,300,3000,30000,300000)
ON CONFLICT DO NOTHING;
"""),

    ("musicians_5a4856 (Notown)", """
CREATE TABLE IF NOT EXISTS musicians_5a4856 (
    ssn  CHAR(10) PRIMARY KEY,
    name VARCHAR(64)
);
INSERT INTO musicians_5a4856 VALUES
  ('001-00-001','John'), ('002-00-002','Paul') ON CONFLICT DO NOTHING;
"""),

    ("instruments_5a4856 (Notown)", """
CREATE TABLE IF NOT EXISTS instruments_5a4856 (
    instrid CHAR(10) PRIMARY KEY,
    dname   VARCHAR(64),
    ikey    VARCHAR(8)
);
INSERT INTO instruments_5a4856 VALUES
  ('GUIT','Guitar','G'), ('BASS','Bass','B') ON CONFLICT DO NOTHING;
"""),

    ("albums_5a4856 (Notown)", """
CREATE TABLE IF NOT EXISTS albums_5a4856 (
    albumid   INTEGER PRIMARY KEY,
    speed     INTEGER,
    copyright DATE,
    title     VARCHAR(128)
);
INSERT INTO albums_5a4856 VALUES
  (1, 33, '1967-06-01', 'Sgt. Pepper'),
  (2, 33, '1969-09-26', 'Abbey Road')
ON CONFLICT DO NOTHING;
"""),

    ("songs_5a4856 (Notown)", """
CREATE TABLE IF NOT EXISTS songs_5a4856 (
    songid INTEGER PRIMARY KEY,
    title  VARCHAR(128),
    author VARCHAR(64)
);
INSERT INTO songs_5a4856 VALUES
  (1,'Come Together','Lennon/McCartney'),
  (2,'Let It Be','Lennon/McCartney')
ON CONFLICT DO NOTHING;
"""),

    ("patient_a78502 (Pharmacy)", """
CREATE TABLE IF NOT EXISTS patient_a78502 (
    ssn     CHAR(11) PRIMARY KEY,
    name    VARCHAR(64),
    age     INTEGER,
    address VARCHAR(128)
);
INSERT INTO patient_a78502 VALUES
  ('111-11-1111','Pat Adams',45,'1 Oak St'),
  ('222-22-2222','Pat Brown',30,'2 Elm Ave')
ON CONFLICT DO NOTHING;
"""),

    ("doctor_a78502 (Pharmacy)", """
CREATE TABLE IF NOT EXISTS doctor_a78502 (
    phy_ssn    CHAR(11) PRIMARY KEY,
    name       VARCHAR(64),
    speciality VARCHAR(64),
    exp_years  INTEGER
);
INSERT INTO doctor_a78502 VALUES
  ('333-33-3333','Dr. Smith','Cardiology',15),
  ('444-44-4444','Dr. Jones','Neurology',10)
ON CONFLICT DO NOTHING;
"""),

    ("pharmacy_a78502 (Pharmacy)", """
CREATE TABLE IF NOT EXISTS pharmacy_a78502 (
    pharm_name VARCHAR(64) PRIMARY KEY,
    address    VARCHAR(128),
    phone_num  VARCHAR(20)
);
INSERT INTO pharmacy_a78502 VALUES
  ('CVS Downtown','3 Main St','555-1234'),
  ('Walgreens','4 Broad St','555-5678')
ON CONFLICT DO NOTHING;
"""),

    ("pharm_co_a78502 (Pharmacy)", """
CREATE TABLE IF NOT EXISTS pharm_co_a78502 (
    co_name   VARCHAR(64) PRIMARY KEY,
    phone_num VARCHAR(20)
);
INSERT INTO pharm_co_a78502 VALUES
  ('PharmaA','555-9000'),
  ('PharmaB','555-9001')
ON CONFLICT DO NOTHING;
"""),

    ("drug_a78502 (Pharmacy)", """
CREATE TABLE IF NOT EXISTS drug_a78502 (
    trade_name VARCHAR(64),
    co_name    VARCHAR(64) REFERENCES pharm_co_a78502(co_name),
    formula    TEXT,
    PRIMARY KEY (trade_name, co_name)
);
INSERT INTO drug_a78502 VALUES
  ('Aspirin','PharmaA','C9H8O4'),
  ('Ibuprofen','PharmaB','C13H18O2')
ON CONFLICT DO NOTHING;
"""),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def init_test_db() -> None:
    dsn = _TEST_DB_URL.replace("postgresql+asyncpg://", "postgresql://")

    print(f"Connecting to test DB: {dsn[:dsn.index('@') + 1]}***")
    try:
        conn = await asyncpg.connect(dsn, timeout=10.0)
    except Exception as e:
        print(f"ERROR: Could not connect to test DB — {e}")
        print("Is the db_test Docker container running?")
        sys.exit(1)

    print(f"Connected. Applying {len(INIT_BLOCKS)} schema/data blocks...\n")

    ok_count = 0
    err_count = 0

    for label, ddl in INIT_BLOCKS:
        # Split on semicolons and execute each statement
        statements = [s.strip() for s in ddl.strip().split(";") if s.strip()]
        try:
            async with conn.transaction():
                for stmt in statements:
                    await conn.execute(stmt)
            print(f"  ✓  {label}")
            ok_count += 1
        except Exception as e:
            print(f"  ✗  {label}: {e}")
            err_count += 1

    await conn.close()

    print(f"\nDone: {ok_count} blocks OK, {err_count} errors.")
    if err_count > 0:
        print("Some blocks failed — review errors above. The rest are still usable.")
    else:
        print("Test DB fully initialised. Ready for SQL harness evaluation.")


if __name__ == "__main__":
    asyncio.run(init_test_db())