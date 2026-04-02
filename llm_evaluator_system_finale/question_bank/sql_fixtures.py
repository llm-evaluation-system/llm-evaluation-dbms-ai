"""
question_bank/sql_fixtures.py — Pre-computed SQL schemas, dummy data, and
execution context for every question in the question bank that requires
PostgreSQL execution to properly evaluate challenger model answers.

Design principles:
  - All table/index/constraint names are GLOBALLY unique (question_id suffix
    applied where name collisions across questions would occur).
  - Each entry may include:
      schema_fixture : str  — DDL to run before the challenger's SQL.
      expected_rows  : list — Expected result tuples for F1 scoring.
      explain_context: str  — EXPLAIN ANALYZE output to inject into the
                              Judge prompt for query-optimisation questions.
      judge_hint     : str  — Free-text hint injected into the Judge prompt
                              (e.g. "The schema for this question is …").
  - ER-diagram questions are EXCLUDED (as per project instructions).
  - Conceptual/theory-only questions that need no DB execution are EXCLUDED.

Usage:
    from question_bank.sql_fixtures import SQL_FIXTURES
    entry = SQL_FIXTURES.get(question_id, {})
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared base schemas reused across multiple questions
# ---------------------------------------------------------------------------

# ── Students / Sailors schemas (used in multiple FOUNDATIONS questions) ──────
_STUDENTS_DDL = """
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
"""

_SAILORS_DDL = """
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
"""

_RESERVES_DDL = """
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
"""

# ── Emp / Works / Dept schema (used in many questions) ──────────────────────
_EMP_WORKS_DEPT_DDL = """
CREATE TABLE IF NOT EXISTS emp (
    eid       INTEGER PRIMARY KEY,
    ename     VARCHAR(64),
    age       INTEGER,
    salary    REAL
);
CREATE TABLE IF NOT EXISTS works (
    eid      INTEGER REFERENCES emp(eid),
    did      INTEGER,
    pct_time INTEGER,
    PRIMARY KEY (eid, did)
);
CREATE TABLE IF NOT EXISTS dept (
    did       INTEGER PRIMARY KEY,
    budget    REAL,
    managerid INTEGER REFERENCES emp(eid)
);
INSERT INTO emp VALUES
  (1, 'Alice',   45, 120000),
  (2, 'Bob',     32, 85000),
  (3, 'Charlie', 55, 150000),
  (4, 'Diana',   28, 60000),
  (5, 'Eve',     41, 95000),
  (6, 'Frank',   35, 75000)
ON CONFLICT DO NOTHING;
INSERT INTO dept VALUES
  (10, 500000, 1),
  (20, 300000, 3),
  (30, 200000, 5)
ON CONFLICT DO NOTHING;
INSERT INTO works VALUES
  (1, 10, 100),
  (2, 10, 50),
  (2, 20, 50),
  (3, 20, 100),
  (4, 30, 100),
  (5, 30, 80),
  (6, 10, 100)
ON CONFLICT DO NOTHING;
"""

# ── Suppliers / Parts / Catalog schema ──────────────────────────────────────
_SUPPLIERS_PARTS_CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS suppliers (
    sid     INTEGER PRIMARY KEY,
    sname   VARCHAR(64),
    address VARCHAR(128)
);
CREATE TABLE IF NOT EXISTS parts (
    pid   INTEGER PRIMARY KEY,
    pname VARCHAR(64),
    color VARCHAR(32)
);
CREATE TABLE IF NOT EXISTS catalog (
    sid  INTEGER REFERENCES suppliers(sid),
    pid  INTEGER REFERENCES parts(pid),
    cost REAL,
    PRIMARY KEY (sid, pid)
);
INSERT INTO suppliers VALUES
  (1, 'Yosemite Sham',   '221 Packer Ave'),
  (2, 'BigBolt Inc',     '500 Main St'),
  (3, 'FastParts Ltd',   '99 Industrial Blvd'),
  (4, 'GreenParts Co',   '10 Green St')
ON CONFLICT DO NOTHING;
INSERT INTO parts VALUES
  (1, 'NutA',   'red'),
  (2, 'BoltB',  'green'),
  (3, 'WasherC','red'),
  (4, 'ScrewD', 'blue'),
  (5, 'NutE',   'green'),
  (6, 'BoltF',  'red')
ON CONFLICT DO NOTHING;
INSERT INTO catalog VALUES
  (1, 1, 10.00),
  (1, 2, 20.00),
  (1, 3, 15.00),
  (2, 1,  8.00),
  (2, 4, 12.00),
  (3, 2, 18.00),
  (3, 5, 22.00),
  (4, 3, 14.00),
  (4, 6, 30.00),
  (1, 5, 25.00),
  (2, 6, 28.00)
ON CONFLICT DO NOTHING;
"""

# ── Flights / Aircraft / Certified / Employees schema ────────────────────────
_FLIGHTS_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS aircraft (
    aid          INTEGER PRIMARY KEY,
    aname        VARCHAR(64),
    cruisingrange INTEGER
);
CREATE TABLE IF NOT EXISTS femployees (
    eid    INTEGER PRIMARY KEY,
    ename  VARCHAR(64),
    salary INTEGER
);
CREATE TABLE IF NOT EXISTS certified (
    eid INTEGER REFERENCES femployees(eid),
    aid INTEGER REFERENCES aircraft(aid),
    PRIMARY KEY (eid, aid)
);
CREATE TABLE IF NOT EXISTS flights (
    flno     INTEGER PRIMARY KEY,
    ffrom    VARCHAR(64),
    fto      VARCHAR(64),
    distance INTEGER,
    departs  TIME,
    arrives  TIME,
    price    INTEGER
);
INSERT INTO aircraft VALUES
  (1, 'Boeing 747',    8000),
  (2, 'Boeing 737',    4000),
  (3, 'Airbus A320',   5500),
  (4, 'Cessna 172',     800),
  (5, 'Boeing 777',   10000)
ON CONFLICT DO NOTHING;
INSERT INTO femployees VALUES
  (1, 'PilotAl',  120000),
  (2, 'PilotBob',  95000),
  (3, 'PilotCal',  75000),
  (4, 'EngDave',   60000),
  (5, 'PilotEve', 110000)
ON CONFLICT DO NOTHING;
INSERT INTO certified VALUES
  (1, 1), (1, 2), (1, 5),
  (2, 2), (2, 3),
  (3, 3), (3, 4),
  (5, 1), (5, 5)
ON CONFLICT DO NOTHING;
INSERT INTO flights VALUES
  (1, 'Madison',    'Chicago',    300, '08:00', '09:30', 250),
  (2, 'Chicago',    'New York',   800, '10:00', '13:00', 450),
  (3, 'Madison',    'New York',   900, '07:00', '11:00', 500),
  (4, 'Bonn',       'Madras',    7500, '12:00', '23:00', 1200),
  (5, 'Madison',    'Los Angeles',2000,'06:00', '09:00', 700),
  (6, 'Los Angeles','Honolulu',  2500, '10:00', '14:00', 800),
  (7, 'New York',   'Madison',    900, '15:00', '19:00', 500),
  (8, 'Chicago',    'Los Angeles',2200,'11:00', '14:30', 650)
ON CONFLICT DO NOTHING;
"""

# ── Student / Class / Enrolled / Faculty schema ──────────────────────────────
_UNIV_ENROLLMENT_DDL = """
CREATE TABLE IF NOT EXISTS student (
    snum  INTEGER PRIMARY KEY,
    sname VARCHAR(64),
    major VARCHAR(64),
    level VARCHAR(8),
    age   INTEGER
);
CREATE TABLE IF NOT EXISTS faculty (
    fid    INTEGER PRIMARY KEY,
    fname  VARCHAR(64),
    deptid INTEGER
);
CREATE TABLE IF NOT EXISTS class (
    name     VARCHAR(64) PRIMARY KEY,
    meets_at TIME,
    room     VARCHAR(16),
    fid      INTEGER REFERENCES faculty(fid)
);
CREATE TABLE IF NOT EXISTS enrolled (
    snum  INTEGER REFERENCES student(snum),
    cname VARCHAR(64) REFERENCES class(name),
    PRIMARY KEY (snum, cname)
);
INSERT INTO student VALUES
  (1001, 'Alice',   'CS',      'JR', 20),
  (1002, 'Bob',     'Math',    'SR', 22),
  (1003, 'Carol',   'CS',      'JR', 21),
  (1004, 'Dave',    'History', 'SO', 19),
  (1005, 'Eve',     'CS',      'JR', 18),
  (1006, 'Frank',   'Physics', 'SR', 20),
  (1007, 'Grace',   'Math',    'JR', 23)
ON CONFLICT (snum) DO UPDATE
  SET sname = EXCLUDED.sname,
      major = EXCLUDED.major,
      level = EXCLUDED.level,
      age   = EXCLUDED.age;
INSERT INTO faculty VALUES
  (10, 'I.Teach',  1),
  (20, 'Prof.Oak', 2),
  (30, 'Dr.Lake',  1)
ON CONFLICT DO NOTHING;
INSERT INTO class VALUES
  ('Databases',   '09:00', 'R128', 10),
  ('Algorithms',  '10:00', 'R200', 20),
  ('OS',          '11:00', 'R128', 10),
  ('Calculus',    '13:00', 'R300', 30),
  ('LinearAlg',   '14:00', 'R128', 30)
ON CONFLICT DO NOTHING;
INSERT INTO enrolled VALUES
  (1001, 'Databases'),
  (1001, 'Algorithms'),
  (1002, 'Databases'),
  (1002, 'Calculus'),
  (1003, 'Databases'),
  (1003, 'OS'),
  (1004, 'Calculus'),
  (1004, 'LinearAlg'),
  (1005, 'Databases'),
  (1005, 'OS'),
  (1006, 'Algorithms'),
  (1006, 'Calculus'),
  (1007, 'Databases'),
  (1007, 'LinearAlg')
ON CONFLICT DO NOTHING;
"""

# ── Emp / Dept for query evaluation and indexing questions ───────────────────
_EMP_DEPT_QE_DDL = """
CREATE TABLE IF NOT EXISTS emp_qe (
    eid   INTEGER PRIMARY KEY,
    did   INTEGER,
    sal   INTEGER,
    hobby VARCHAR(32)
);
CREATE TABLE IF NOT EXISTS dept_qe (
    did    INTEGER PRIMARY KEY,
    dname  VARCHAR(32),
    floor  INTEGER,
    budget REAL
);
INSERT INTO emp_qe VALUES
  (1,  10, 55000, 'tennis'),
  (2,  10, 62000, 'yodeling'),
  (3,  20, 48000, 'chess'),
  (4,  20, 71000, 'tennis'),
  (5,  30, 58000, 'yodeling'),
  (6,  30, 44000, 'chess'),
  (7,  10, 39000, 'golf'),
  (8,  20, 83000, 'yodeling'),
  (9,  30, 95000, 'tennis'),
  (10, 10, 27000, 'chess')
ON CONFLICT DO NOTHING;
INSERT INTO dept_qe VALUES
  (10, 'Engineering', 1, 400000),
  (20, 'Marketing',   2, 250000),
  (30, 'Finance',     1, 300000)
ON CONFLICT DO NOTHING;
"""

# ── Emp/Dept for BCNF/Indexing questions (different naming) ─────────────────
_EMP_DEPT_IDX_DDL = """
CREATE TABLE IF NOT EXISTS emp_idx (
    eid     INTEGER PRIMARY KEY,
    ename   VARCHAR(64),
    sal     INTEGER,
    age     INTEGER,
    did     INTEGER
);
CREATE TABLE IF NOT EXISTS dept_idx (
    did       INTEGER PRIMARY KEY,
    budget    INTEGER,
    floor     INTEGER,
    mgr_eid   INTEGER
);
INSERT INTO emp_idx VALUES
  (1, 'Alice',   90000, 45, 10),
  (2, 'Bob',     55000, 32, 10),
  (3, 'Charlie', 78000, 55, 20),
  (4, 'Diana',   42000, 28, 20),
  (5, 'Eve',     110000, 41, 30),
  (6, 'Frank',   63000, 35, 30),
  (7, 'Grace',   88000, 50, 10),
  (8, 'Hank',    15000, 25, 20)
ON CONFLICT DO NOTHING;
INSERT INTO dept_idx VALUES
  (10, 500000, 1, 1),
  (20, 300000, 3, 3),
  (30, 200000, 10, 5)
ON CONFLICT DO NOTHING;
"""

# ── Executives table (used in projection sorting question) ───────────────────
_EXECUTIVES_DDL = """
CREATE TABLE IF NOT EXISTS executives (
    ename   VARCHAR(64),
    title   VARCHAR(64),
    dname   VARCHAR(64),
    address VARCHAR(128)
);
INSERT INTO executives VALUES
  ('Alice Smith',  'CEO',   'Engineering', '100 Main St'),
  ('Bob Jones',    'CTO',   'Engineering', '200 Elm St'),
  ('Carol White',  'CFO',   'Finance',     '300 Oak Ave'),
  ('Dave Brown',   'COO',   'Operations',  '400 Pine Rd'),
  ('Eve Davis',    'VP',    'Marketing',   '500 Maple Dr'),
  ('Frank Miller', 'VP',    'Engineering', '600 Cedar Ln')
ON CONFLICT DO NOTHING;
"""

# ── Finance table (used in join order question) ──────────────────────────────
_FINANCE_DDL = """
CREATE TABLE IF NOT EXISTS finance (
    did      INTEGER PRIMARY KEY,
    budget   REAL,
    sales    REAL,
    expenses REAL
);
INSERT INTO finance VALUES
  (10, 400000, 900000, 500000),
  (20, 250000, 600000, 350000),
  (30, 300000, 750000, 400000)
ON CONFLICT DO NOTHING;
"""

# ── Parts table (for DB design / indexing) ───────────────────────────────────
_PARTS_DDL = """
CREATE TABLE IF NOT EXISTS parts_design (
    pid       INTEGER PRIMARY KEY,
    pname     VARCHAR(64),
    cost      REAL,
    num_avail INTEGER
);
INSERT INTO parts_design VALUES
  (1, 'nuts',   2.50, 1000),
  (2, 'bolts',  1.80,  500),
  (3, 'washer', 0.50, 2000),
  (4, 'screw',  0.30, 3000),
  (5, 'nail',   0.10, 5000)
ON CONFLICT DO NOTHING;
"""

# ── NumReservations view (ADDITIONAL TOPICS - views question) ────────────────
_NUM_RESERVATIONS_DDL = _SAILORS_DDL + _RESERVES_DDL + """
CREATE OR REPLACE VIEW numreservations AS
    SELECT s.sid, s.sname, COUNT(*) AS numres
    FROM sailors s
    JOIN reserves r ON s.sid = r.sid
    GROUP BY s.sid, s.sname;
"""

# ── Sales / Times (SQL Windowing question) ───────────────────────────────────
_SALES_TIMES_DDL = """
CREATE TABLE IF NOT EXISTS times_dim (
    timeid INTEGER PRIMARY KEY,
    year   INTEGER,
    month  INTEGER,
    day    INTEGER
);
CREATE TABLE IF NOT EXISTS sales_fact (
    saleid INTEGER PRIMARY KEY,
    timeid INTEGER REFERENCES times_dim(timeid),
    sales  REAL
);
INSERT INTO times_dim VALUES
  (1, 2020, 1, 15),
  (2, 2020, 6, 20),
  (3, 2021, 3, 10),
  (4, 2021, 9, 25),
  (5, 2022, 2, 28)
ON CONFLICT DO NOTHING;
INSERT INTO sales_fact VALUES
  (1, 1, 100000),
  (2, 2, 150000),
  (3, 3, 120000),
  (4, 4, 180000),
  (5, 5, 200000)
ON CONFLICT DO NOTHING;
"""

# ── Professors / Depts / Projects / Graduates (university DB from Ex 2.3/3.13) ─
_UNIVERSITY_DB_DDL = """
CREATE TABLE IF NOT EXISTS professors (
    prof_ssn  CHAR(10) PRIMARY KEY,
    name      VARCHAR(64),
    age       INTEGER,
    rank      INTEGER,
    speciality VARCHAR(64)
);
CREATE TABLE IF NOT EXISTS depts (
    dno    INTEGER PRIMARY KEY,
    dname  VARCHAR(64),
    office VARCHAR(16)
);
CREATE TABLE IF NOT EXISTS projects (
    pid        INTEGER PRIMARY KEY,
    sponsor    VARCHAR(64),
    start_date DATE,
    end_date   DATE,
    budget     REAL
);
CREATE TABLE IF NOT EXISTS graduates (
    grad_ssn VARCHAR(10) PRIMARY KEY,
    name     VARCHAR(64),
    age      INTEGER,
    deg_prog VARCHAR(32)
);
INSERT INTO professors VALUES
  ('111-22-3333', 'Prof. Adams', 52, 3, 'Databases'),
  ('222-33-4444', 'Prof. Baker', 45, 2, 'AI'),
  ('333-44-5555', 'Prof. Clark', 38, 1, 'Networks')
ON CONFLICT DO NOTHING;
INSERT INTO depts VALUES
  (1, 'Computer Science', 'A101'),
  (2, 'Mathematics',      'B202'),
  (3, 'Physics',          'C303')
ON CONFLICT DO NOTHING;
INSERT INTO projects VALUES
  (101, 'NSF', '2022-01-01', '2024-12-31', 500000),
  (102, 'DARPA', '2021-06-01', '2023-05-31', 1200000),
  (103, 'NIH', '2023-03-01', '2025-02-28', 750000)
ON CONFLICT DO NOTHING;
INSERT INTO graduates VALUES
  ('444-55-6666', 'Grad Ann',  26, 'PhD'),
  ('555-66-7777', 'Grad Bill', 24, 'MS'),
  ('666-77-8888', 'Grad Cara', 27, 'PhD')
ON CONFLICT DO NOTHING;
"""

# ── Emp for DB Design/Tuning questions (separate from above) ─────────────────
_EMP_DEPT_BCNF_DDL = """
CREATE TABLE IF NOT EXISTS emp_bcnf (
    eid    INTEGER PRIMARY KEY,
    ename  VARCHAR(64),
    addr   VARCHAR(128),
    sal    REAL,
    age    INTEGER,
    yrs    INTEGER,
    deptid INTEGER
);
CREATE TABLE IF NOT EXISTS dept_bcnf (
    did       INTEGER PRIMARY KEY,
    dname     VARCHAR(64),
    floor     INTEGER,
    budget    REAL
);
INSERT INTO emp_bcnf VALUES
  (1,  'Alice',   '1 Main St',    120000, 45, 20, 10),
  (2,  'Bob',     '2 Elm St',      85000, 32, 8,  10),
  (3,  'Charlie', '3 Oak Ave',    150000, 55, 30, 20),
  (4,  'Diana',   '4 Pine Rd',     60000, 28, 3,  20),
  (5,  'Eve',     '5 Maple Dr',    95000, 41, 15, 30),
  (6,  'Frank',   '6 Cedar Ln',    75000, 35, 10, 30),
  (7,  'Grace',   '7 Birch Ct',   110000, 50, 22, 10),
  (8,  'Hank',    '8 Spruce Pl',   40000, 26, 2,  20)
ON CONFLICT DO NOTHING;
INSERT INTO dept_bcnf VALUES
  (10, 'Engineering',  1, 500000),
  (20, 'Marketing',    2, 300000),
  (30, 'Finance',      3, 200000)
ON CONFLICT DO NOTHING;
"""

# ── Dept_manager / Emp for security question ─────────────────────────────────
_DEPT_MANAGER_DDL = """
CREATE TABLE IF NOT EXISTS dept_mgr (
    did       INTEGER PRIMARY KEY,
    dname     VARCHAR(64),
    location  VARCHAR(64),
    managerid INTEGER
);
CREATE TABLE IF NOT EXISTS emp_mgr (
    eid  INTEGER PRIMARY KEY,
    sal  REAL
);
INSERT INTO emp_mgr VALUES
  (1, 120000), (2, 85000), (3, 150000),
  (4, 60000),  (5, 95000)
ON CONFLICT DO NOTHING;
INSERT INTO dept_mgr VALUES
  (10, 'Engineering', 'NYC',  1),
  (20, 'Marketing',   'NYC',  3),
  (30, 'Finance',     'SF',   5),
  (40, 'Operations',  'SF',   2),
  (50, 'R&D',         'Boston', 4)
ON CONFLICT DO NOTHING;
"""


# ---------------------------------------------------------------------------
# Main fixture registry
# ---------------------------------------------------------------------------

SQL_FIXTURES: dict[str, dict] = {

    # =========================================================================
    # FOUNDATIONS — SQL / Practical questions
    # =========================================================================

    # ID: 41de485a98af  — Physical/logical data independence (conceptual, no exec)
    "41de485a98af": {
        "judge_hint": (
            "This is a conceptual question about data independence in the relational "
            "model. No SQL execution is required. Evaluate the answer based on whether "
            "it correctly explains physical independence (SQL users don't know physical "
            "layout) and logical independence (views hide conceptual schema changes)."
        ),
    },

    # ID: c6c4ce7444f2 — Students with age < 18, modify query
    "c6c4ce7444f2": {
        "schema_fixture": _STUDENTS_DDL,
        "expected_rows": [
            {"login": "madayan@music"},
            {"login": "guldu@music"},
        ],
        "judge_hint": (
            "Schema: students(sid, sname, login, age, gpa). "
            "The instance contains 6 rows. Age < 18 matches sid=53831 (Madayan, age=11) "
            "and sid=53832 (Guldu, age=12). Part 1 asks for only the login column. "
            "Part 2 adds gpa >= 2: only Guldu (gpa=2.0) qualifies — Madayan (gpa=1.8) is excluded."
        ),
    },

    # ID: 7f30162b5f47 — Ternary relationship R between A, B, C → CREATE TABLE SQL
    "7f30162b5f47": {
        "schema_fixture": """
CREATE TABLE IF NOT EXISTS rel_a_7f30 (
    a1 CHAR(10) PRIMARY KEY,
    a2 CHAR(10)
);
CREATE TABLE IF NOT EXISTS rel_b_7f30 (
    b1 CHAR(10) PRIMARY KEY,
    b2 CHAR(10)
);
CREATE TABLE IF NOT EXISTS rel_c_7f30 (
    c1 CHAR(10) PRIMARY KEY,
    c2 CHAR(10)
);
CREATE TABLE IF NOT EXISTS rel_r_7f30 (
    a1 CHAR(10) PRIMARY KEY,
    b1 CHAR(10) UNIQUE,
    c1 CHAR(10),
    FOREIGN KEY (a1) REFERENCES rel_a_7f30(a1),
    FOREIGN KEY (b1) REFERENCES rel_b_7f30(b1),
    FOREIGN KEY (c1) REFERENCES rel_c_7f30(c1)
);
""",
        "expected_rows": [],
        "judge_hint": (
            "This question asks to write CREATE TABLE statements for a ternary "
            "relationship R between A, B, C where A has key constraint + total "
            "participation, and B has key constraint. The key insight: A's primary key "
            "is a1 (with UNIQUE on b1 for B's key constraint). The relationship R is "
            "merged into A's table since A has total participation + key constraint."
        ),
    },

    # ID: 53f9113f9138 — University database CREATE TABLE (Exercise 3.13)
    "53f9113f9138": {
        "schema_fixture": _UNIVERSITY_DB_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Professors(prof_ssn, name, age, rank, speciality), "
            "Depts(dno, dname, office), Projects(pid, sponsor, start_date, end_date, budget), "
            "Graduates(grad_ssn, name, age, deg_prog). "
            "Question asks to write SQL CREATE TABLE statements preserving constraints. "
            "FK relationships: Professors work_in Depts, Professors work_on Projects, "
            "Graduates are advised by Professors."
        ),
    },

    # ID: 5a4856d130ba — Notown database CREATE TABLE (Exercise 3.15)
    "5a4856d130ba": {
        "schema_fixture": """
CREATE TABLE IF NOT EXISTS musicians_5a4856 (
    ssn  CHAR(10) PRIMARY KEY,
    name VARCHAR(64)
);
CREATE TABLE IF NOT EXISTS instruments_5a4856 (
    instrid CHAR(10) PRIMARY KEY,
    dname   VARCHAR(64),
    ikey    VARCHAR(8)
);
CREATE TABLE IF NOT EXISTS albums_5a4856 (
    albumid   INTEGER PRIMARY KEY,
    speed     INTEGER,
    copyright DATE,
    title     VARCHAR(128)
);
CREATE TABLE IF NOT EXISTS songs_5a4856 (
    songid  INTEGER PRIMARY KEY,
    title   VARCHAR(128),
    author  VARCHAR(64)
);
""",
        "expected_rows": [],
        "judge_hint": (
            "Notown schema: Musicians(ssn, name), Instruments(instrId, dname, key), "
            "Albums(albumIdentifier, speed, copyrightDate, title), Songs(songId, title, author). "
            "Question asks for CREATE TABLE SQL. Pay attention to weak entity sets "
            "(Phone, Place) and how relationships are mapped to tables."
        ),
    },

    # ID: a78502b807ea — Prescriptions-R-X pharmacy CREATE TABLE (Exercise 3.17)
    "a78502b807ea": {
        "schema_fixture": """
CREATE TABLE IF NOT EXISTS patient_a78502 (
    ssn     CHAR(11) PRIMARY KEY,
    name    VARCHAR(64),
    age     INTEGER,
    address VARCHAR(128)
);
CREATE TABLE IF NOT EXISTS doctor_a78502 (
    phy_ssn    CHAR(11) PRIMARY KEY,
    name       VARCHAR(64),
    speciality VARCHAR(64),
    exp_years  INTEGER
);
CREATE TABLE IF NOT EXISTS pharmacy_a78502 (
    pharm_name VARCHAR(64) PRIMARY KEY,
    address    VARCHAR(128),
    phone_num  VARCHAR(20)
);
CREATE TABLE IF NOT EXISTS pharm_co_a78502 (
    co_name   VARCHAR(64) PRIMARY KEY,
    phone_num VARCHAR(20)
);
CREATE TABLE IF NOT EXISTS drug_a78502 (
    trade_name VARCHAR(64),
    co_name    VARCHAR(64) REFERENCES pharm_co_a78502(co_name),
    formula    TEXT,
    PRIMARY KEY (trade_name, co_name)
);
""",
        "expected_rows": [],
        "judge_hint": (
            "Pharmacy schema: Patient(ssn, name, age, address), Doctor(phy_ssn, name, "
            "speciality, exp_years), Pharmacy(name, address, phone_num), "
            "Pharm_co(name, phone_num), Drug(weak entity: trade_name, formula). "
            "Answer should write CREATE TABLE statements with correct PKs and FKs."
        ),
    },

    # ID: 0c053e7bed52 — SeniorEmp view, updatable views (Exercise 3.19)
    "0c053e7bed52": {
        "schema_fixture": _EMP_WORKS_DEPT_DDL + """
CREATE OR REPLACE VIEW senioremp AS
    SELECT e.ename AS sname, e.age AS sage, e.salary
    FROM emp e
    WHERE e.age > 50;
""",
        "expected_rows": [
            {"sname": "Charlie"},
        ],
        "judge_hint": (
            "Schema: Emp(eid, ename, age, salary), Works(eid, did, pct_time), "
            "Dept(did, budget, managerid). View SeniorEmp selects ename,age,salary "
            "where age > 50. The query SELECT S.sname FROM SeniorEmp S WHERE "
            "S.salary > 100000 should return employees over 50 with salary > 100k. "
            "Charlie (age=55, salary=150000) and Alice (age=45) — only Charlie matches."
        ),
    },

    # ID: c36cf0251982 — Student/Class/Enrolled/Faculty queries (Exercise 5.1)
    "c36cf0251982": {
        "schema_fixture": _UNIV_ENROLLMENT_DDL,
        "expected_rows": [
            {"sname": "Alice"},
            {"sname": "Carol"},
            {"sname": "Eve"},
            {"sname": "Grace"},
        ],
        "judge_hint": (
            "Schema: Student(snum,sname,major,level,age), Class(name,meets_at,room,fid), "
            "Enrolled(snum,cname), Faculty(fid,fname,deptid). "
            "Question 1: JR students enrolled in a class taught by 'I.Teach' (fid=10). "
            "I.Teach teaches: Databases(fid=10) and OS(fid=10). "
            "JR students: Alice(1001), Carol(1003), Eve(1005), Grace(1007). "
            "Enrolled in I.Teach's classes: Alice→Databases, Carol→Databases+OS, "
            "Eve→Databases+OS, Grace→Databases. All 4 are JR. "
            "Answer: Alice, Carol, Eve, Grace. "
            "expected_rows are for query 1 only."
        ),
    },

    # ID: 0def7bdc534e — Flights/Aircraft/Certified/Employees SQL (Exercise 5.3)
    "0def7bdc534e": {
        "schema_fixture": _FLIGHTS_SCHEMA_DDL,
        "expected_rows": [
            {"aname": "Boeing 747"},
            {"aname": "Boeing 737"},
            {"aname": "Boeing 777"},
        ],
        "judge_hint": (
            "EVALUATION INSTRUCTIONS — READ BEFORE SCORING:\n"
            "The test database uses these EXACT table names (renamed to avoid conflicts):\n"
            "  'Employees' in the question = 'femployees' in the test DB\n"
            "  column 'from' in the question = 'ffrom' in the test DB\n"
            "  column 'to' in the question = 'fto' in the test DB\n"
            "DO NOT penalise the model for using femployees, ffrom, or fto — "
            "these are CORRECT. A model using femployees/ffrom/fto has followed "
            "the schema instructions correctly. Flag it as CORRECT, not an error.\n"
            "SQL execution result (F1 score) is authoritative for query 1 correctness.\n"
            "Query 1 correct answer: Boeing 747 (aid=1, pilots Al 120k+Eve 110k), "
            "Boeing 737 (aid=2, pilots Al 120k+Bob 95k), "
            "Boeing 777 (aid=5, pilots Al 120k+Eve 110k). "
            "Cessna172 and AirbusA320 have pilots earning <=80k so are excluded."
        ),
    },

    # ID: 6c3f53eb80bb — Sailors AVG/SUM/COUNT queries (Exercise 5.5)
    "6c3f53eb80bb": {
        "schema_fixture": _SAILORS_DDL,
        "expected_rows": [
            {"avg": 5.333333},
        ],
        "judge_hint": (
            "Schema: Sailors(sid, sname, rating, age). Data: jones(3), jonah(6), "
            "ahab(7), moby(NULL). "
            "AVG(rating): ignores NULL → (3+6+7)/3 = 5.33. "
            "SUM(rating): ignores NULL → 16. "
            "COUNT(rating): ignores NULL → 3. COUNT(*) → 4. "
            "Key point: NULL rating for 'moby' is excluded from AVG/SUM/COUNT(rating) "
            "but included in COUNT(*)."
        ),
    },

    # ID: 6bd4ccd25d85 — Trigger mechanism discussion (conceptual + SQL)
    "6bd4ccd25d85": {
        "judge_hint": (
            "This is a conceptual question comparing SQL triggers vs integrity constraints. "
            "No DB execution needed. Key points: triggers can perform complex actions "
            "(INSERT/UPDATE/DELETE/DDL) that constraints cannot; triggers execute "
            "before/after modifications; constraints are simpler, better optimized, "
            "and easier to understand. If the model writes trigger syntax, validate it "
            "is correct PL/pgSQL."
        ),
    },

    # ID: c62a41f94461 — Emp/Works/Dept table constraints and assertions (Exercise 5.7)
    "c62a41f94461": {
        "schema_fixture": _EMP_WORKS_DEPT_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Emp(eid,ename,age,salary), Works(eid,did,pct_time), "
            "Dept(did,budget,managerid). Question asks to: "
            "1) Add CHECK(salary >= 10000) to Emp. "
            "2) Define table constraint on Dept ensuring managers have age > 30. "
            "3) Define equivalent CREATE ASSERTION. "
            "4) DELETE employees making more than their manager. "
            "Evaluate correctness of the SQL DDL and DML statements."
        ),
    },

    # =========================================================================
    # APPLICATION DEVELOPMENT (mostly conceptual)
    # =========================================================================

    "2ca79e49a73b": {
        "judge_hint": (
            "Conceptual question about Cursor, Embedded SQL, JDBC, SQLJ, stored procedures. "
            "No DB execution needed. Evaluate based on correctness of definitions and "
            "explanation of differences between JDBC (dynamic, runtime binding) and "
            "SQLJ (static, compile-time binding)."
        ),
    },

    "4d30f627adcf": {
        "judge_hint": (
            "Conceptual question about JDBC and SQLJ connection/transaction/stored procedure APIs. "
            "No DB execution needed. Key: JDBC uses DriverManager.getConnection(), "
            "Connection.setAutoCommit(false)/commit()/rollback(), CallableStatement. "
            "SQLJ uses #sql { CALL proc() } syntax."
        ),
    },

    "e1330b9f31d3": {
        "judge_hint": (
            "Conceptual question about exception handling in embedded SQL (SQLSTATE/SQLCODE), "
            "dynamic SQL, JDBC (try/catch SQLException), and SQLJ. "
            "No DB execution needed."
        ),
    },

    # =========================================================================
    # STORAGE AND INDEXING
    # =========================================================================

    "7214203d8a10": {
        "schema_fixture": _EMP_DEPT_IDX_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Emp(eid,ename,sal,age,did), Dept(did,budget,floor,mgr_eid). "
            "Salaries 10k-100k, ages 20-80, ~5 employees/dept, 10 floors, budgets 10k-1M. "
            "Question asks which index to choose for two queries. "
            "Q1 (print ename,age,sal for all employees): unclustered hash on (ename,age,sal) "
            "allows index-only scan; without index-only plans → no index (full scan). "
            "Q2 (dids of depts on floor 10 with budget < 15000): clustered B+ tree on "
            "(floor,budget) is best — allows efficient range scan on both predicates."
        ),
    },

    "45e705cfe771": {
        "judge_hint": (
            "Mathematical/conceptual question about B+ tree node capacity. "
            "No DB execution needed. "
            "Formula: entries_per_node = floor(P / (K + Pr)). "
            "Fanout ≈ floor(P / (K + Pr)). "
            "Height ≈ ceil(log_fanout(N))."
        ),
    },

    "de8d75416edf": {
        "schema_fixture": _EMP_DEPT_BCNF_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Emp([eid], ename, addr, sal, age, yrs, deptid), "
            "Dept([did], dname, floor, budget). "
            "Question asks to design physical schema (indexes) for 6 queries. "
            "Key recommendations: "
            "- dense unclustered B+ on (age,sal) for index-only scan (query e). "
            "- unclustered B+ on deptid of Emp + unclustered on (dname,did) of Dept (query b). "
            "- unclustered on ename of Emp (query c). "
            "- clustered B+ on floor of Dept (query f). "
            "- dense unclustered B+ on sal (query d — average salary)."
        ),
    },

    "8fdae0f06d9c": {
        "schema_fixture": _EMP_DEPT_BCNF_DDL + """
CREATE OR REPLACE VIEW mgrAge AS
    SELECT d.dname, e.age
    FROM emp_bcnf e
    JOIN dept_bcnf d ON d.did = e.eid;
""",
        "expected_rows": [],
        "judge_hint": (
            "Schema: Emp(eid,ename,addr,sal,age,yrs,deptid), Dept(did,dname,floor,budget), "
            "View MgrAge(dname,age) = manager ages by department. "
            "Question compares nested query vs view-based query performance. "
            "The judge should evaluate the correctness of the performance analysis: "
            "nested query wins when few/no employees have sal > 100k (short-circuit); "
            "view query wins when many employees have sal > 100k and Dept is large."
        ),
    },

    "85e72f20267a": {
        "schema_fixture": _PARTS_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Parts([pid], pname, cost, num_avail). "
            "Two important queries: "
            "1) Total num_avail by pname (GROUP BY pname). "
            "2) pids of parts with highest cost. "
            "Physical design: heap file + dense unclustered B+ on (pname,num_avail) "
            "for query 1 index-only scan; dense unclustered B+ on (cost,pid) for query 2. "
            "For schema redesign: vertical partition into Parts1(pid,cost) and "
            "Parts2(pid,pname,num_avail) with clustered indexes."
        ),
    },

    # =========================================================================
    # QUERY EVALUATION
    # =========================================================================

    "2b3771b3253f": {
        "schema_fixture": _EMP_DEPT_QE_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Emp(eid,did,sal,hobby), Dept(did,dname,floor,budget). "
            "Question asks which attributes must be examined for 4 queries: "
            "1) SELECT * FROM Emp → all Emp attrs. "
            "2) SELECT * FROM Emp, Dept → all attrs of both. "
            "3) SELECT * FROM Emp E, Dept D WHERE E.did=D.did → all attrs of both. "
            "4) SELECT E.eid, D.dname FROM Emp E, Dept D WHERE E.did=D.did → "
            "   E.eid, E.did (for join), D.did (for join), D.dname."
        ),
    },

    "d4ed97724281": {
        "schema_fixture": _EXECUTIVES_DDL,
        "expected_rows": [
            {"title": "CEO", "ename": "Alice Smith"},
            {"title": "CFO", "ename": "Carol White"},
            {"title": "COO", "ename": "Dave Brown"},
            {"title": "CTO", "ename": "Bob Jones"},
            {"title": "VP",  "ename": "Eve Davis"},
            {"title": "VP",  "ename": "Frank Miller"},
        ],
        "judge_hint": (
            "Schema: Executives(ename,title,dname,address). "
            "Query: SELECT DISTINCT E.title, E.ename FROM Executives E. "
            "Question focuses on cost analysis of sorting-based projection. "
            "10000 pages, 10 buffer pages. First pass: 10000/10 = 1000 runs. "
            "Additional merge passes with 10 buffers: ceil(log_9(1000)) = 3 passes. "
            "The judge should evaluate I/O cost calculations and index alternatives."
        ),
        "explain_context": (
            "EXPLAIN ANALYZE SELECT DISTINCT title, ename FROM executives;\n"
            "→ Sort (cost=45.41..50.16 rows=1900 width=128)\n"
            "    Sort Key: title, ename\n"
            "    Sort Method: quicksort\n"
            "  -> Seq Scan on executives (cost=0.00..32.90 rows=1890 width=128)\n"
            "This shows the optimizer choosing sequential scan + sort for DISTINCT projection."
        ),
    },

    "d9c7e0be6bc3": {
        "schema_fixture": _EMP_DEPT_QE_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Conceptual question about query optimization. Key points: "
            "1) SQL query block: single SELECT-FROM-WHERE (no nesting). "
            "2) Reduction factor: fraction of tuples surviving a predicate. "
            "3) Projection before selection: when inner relation of nested loop is small "
            "   and projection reduces page count significantly. "
            "4) Using unclustered B+ indexes for sort-merge join: good when few "
            "   tuples/page (one page per tuple), bad when many tuples/page (repeated reads). "
            "5) Interesting orders in System R: orderings useful for future operations "
            "   (ORDER BY, GROUP BY, join) kept even if not cheapest for current step."
        ),
    },

    "4e7a245643d9": {
        "schema_fixture": _EMP_DEPT_QE_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Emp(eid,did,sal,hobby), Dept(did,dname,floor,budget). "
            "Attributes examined for 6 queries: "
            "1) SELECT COUNT(*) FROM Emp E, Dept D WHERE E.did=D.did → E.did, D.did. "
            "2) SELECT MAX(E.sal) ... WHERE E.did=D.did → E.sal, E.did, D.did. "
            "3) ... AND D.floor=5 → E.sal, E.did, D.did, D.floor. "
            "4) SELECT E.did, COUNT(*) ... GROUP BY D.did → E.did, D.did. "
            "5) SELECT D.floor, AVG(D.budget) ... GROUP BY D.floor HAVING COUNT(*)>2 "
            "   → D.floor, D.budget. "
            "6) ... ORDER BY D.floor → D.floor, D.budget."
        ),
    },

    "5893d14b3015": {
        "schema_fixture": """
CREATE TABLE IF NOT EXISTS rel_r_5893 (
    a INTEGER,
    b INTEGER,
    c_attr INTEGER,
    d_attr INTEGER,
    PRIMARY KEY (a)
);
CREATE TABLE IF NOT EXISTS rel_s_5893 (
    c_attr INTEGER PRIMARY KEY,
    d_attr INTEGER,
    e_attr INTEGER,
    f_attr INTEGER,
    g_attr INTEGER
);
INSERT INTO rel_r_5893 VALUES
  (1, 10, 1, 100), (2, 20, 2, 200), (3, 30, 3, 300),
  (4, 40, 1, 400), (5, 50, 2, 500)
ON CONFLICT DO NOTHING;
INSERT INTO rel_s_5893 VALUES
  (1, 100, 1000, 10000, 100000),
  (2, 200, 2000, 20000, 200000),
  (3, 300, 3000, 30000, 300000)
ON CONFLICT DO NOTHING;
""",
        "expected_rows": [],
        "judge_hint": (
            "Question about I/O cost of projection + join alternatives. "
            "R: 10 pages (300B tuples), S: 100 pages (500B tuples), page=1024B. "
            "C is key for S, A is key for R, each S tuple joins exactly one R tuple. "
            "Result: A,B,C,D (450B). ~2 records/page → 100 pages result. "
            "With 3 buffer pages, SNL join: outer=R(10p), each R page reads all S(100p). "
            "Cost = 10 + 10×100 = 1010 I/Os for join-then-project-on-the-fly."
        ),
    },

    "5f979c3f0e84": {
        "schema_fixture": _EMP_DEPT_QE_DDL + _FINANCE_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Emp(eid,did,sal,hobby), Dept(did,dname,floor,phone), "
            "Finance(did,budget,sales,expenses). "
            "Query filters: D.floor=1, E.sal>=59000, E.hobby='yodeling'. "
            "Emp: 50000 tuples. Dept: 5000 tuples. Finance: 5000 tuples. "
            "RF for sal>=59000: 1/50. RF for hobby='yodeling': 1/200. "
            "Estimated Emp tuples after filters: 50000×(1/50)×(1/200) = 5. "
            "Best join order: filter Emp first (5 tuples), then index-NL join to Dept, "
            "then index-NL join to Finance."
        ),
    },

    # =========================================================================
    # TRANSACTION MANAGEMENT
    # =========================================================================

    "f25e0c1cd0e0": {
        "schema_fixture": _UNIV_ENROLLMENT_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Student(snum,sname,major,level,age), Class(name,meets_at,room,fid), "
            "Enrolled(snum,cname), Faculty(fid,fname,deptid). "
            "Question asks appropriate SQL isolation level for 4 operations: "
            "1) Insert new enrollment → READ UNCOMMITTED (no existing row lock needed). "
            "2) Change enrollment → READ COMMITTED (update one existing row). "
            "3) Assign faculty to class with least students → SERIALIZABLE (phantom problem). "
            "4) COUNT students per class → SERIALIZABLE (phantom problem)."
        ),
    },

    "76579c82c472": {
        "schema_fixture": _SUPPLIERS_PARTS_CATALOG_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Suppliers(sid,sname,address), Parts(pid,pname,color), "
            "Catalog(sid,pid,cost). "
            "Question asks to show database instances and SQL statements where "
            "isolation level differences are observable: "
            "1) SERIALIZABLE vs REPEATABLE READ: phantom problem (INSERT visible in RR). "
            "2) REPEATABLE READ vs READ COMMITTED: unrepeatable read (UPDATE visible). "
            "3) READ COMMITTED vs READ UNCOMMITTED: dirty read (uncommitted update visible)."
        ),
    },

    # =========================================================================
    # DATABASE DESIGN AND TUNING
    # =========================================================================

    "165c1883b2d7": {
        "judge_hint": (
            "Conceptual security question about Bell-LaPadula model, covert channels, "
            "polyinstantiation, mandatory vs discretionary access controls, encryption. "
            "No DB execution needed. Evaluate based on correctness of explanations "
            "for each of the 12 sub-questions."
        ),
    },

    "ed3b91e88cb8": {
        "schema_fixture": _DEPT_MANAGER_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Dept([did], dname, location, managerid), Emp([eid], sal). "
            "Two important queries: "
            "1) List names/ids of managers for each dept in a location (alpha by dname). "
            "2) Average salary of managers in a location. "
            "Physical design: clustered B+ tree on (location,dname) of Dept; "
            "hash index on eid of Emp. "
            "Without indexes: horizontal decomposition of Dept by location; "
            "sorted file organizations."
        ),
    },

    # ID: c0d653bd3abd — SQL isolation levels + locking protocols (Exercise 17.9)
    "c0d653bd3abd": {
        "judge_hint": (
            "Conceptual question about SQL isolation levels × access modes (8 combinations). "
            "No DB execution needed. "
            "Key locking protocols per class: "
            "SERIALIZABLE+RO: Strict 2PL + predicate locks, no X locks. "
            "SERIALIZABLE+RW: Strict 2PL + predicate locks. "
            "REPEATABLE READ+RO: Strict 2PL on individual objects, no X locks, no predicate locks. "
            "REPEATABLE READ+RW: Strict 2PL on individual objects. "
            "READ COMMITTED+RO: S locks acquired and released immediately. "
            "READ COMMITTED+RW: X locks held to EOT; S locks released immediately. "
            "READ UNCOMMITTED+RO: No S locks at all. "
            "READ UNCOMMITTED+RW: X locks held to EOT; no S locks. "
            "Schedules with all SERIALIZABLE → conflict-serializable, serializable, recoverable. "
            "Schedules with all READ ONLY → conflict-serializable, serializable, recoverable."
        ),
    },


    # ID: 91d0bd09fe08 — Suppliers/Parts/Catalog: RA/TRC/DRC/SQL for 12 queries (Exercise 4.3)
    "91d0bd09fe08": {
        "schema_fixture": _SUPPLIERS_PARTS_CATALOG_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Suppliers(sid,sname,address), Parts(pid,pname,color), "
            "Catalog(sid,pid,cost). "
            "Question asks for RA, TRC, DRC, and SQL for 12 queries. "
            "Key data: colors are 'red'/'green'/'blue'; address '221 Packer Ave' for sid=1. "
            "Queries requiring division (5,6,7,8): use NOT EXISTS double negation in SQL. "
            "Query 11 (most expensive parts from Yosemite Sham): use >= ALL subquery. "
            "Query 12 (supplied by every supplier at < $200): division + cost filter. "
            "Evaluate RA notation correctness (π, σ, ⋈, /, ρ, ∪, ∩, −), "
            "TRC/DRC formula structure, and SQL equivalence."
        ),
    },

    "8c61735b507d": {
        "schema_fixture": _FLIGHTS_SCHEMA_DDL,
        "expected_rows": [],
        "judge_hint": (
            "EVALUATION INSTRUCTIONS — READ BEFORE SCORING:\n"
            "The test database uses these EXACT table names (renamed to avoid conflicts):\n"
            "  'Employees' in the question = 'femployees' in the test DB\n"
            "  column 'from' in the question = 'ffrom' in the test DB\n"
            "  column 'to' in the question = 'fto' in the test DB\n"
            "DO NOT penalise the model for using femployees, ffrom, or fto — "
            "these are CORRECT. A model using femployees/ffrom/fto has followed "
            "the schema instructions correctly. Flag it as CORRECT, not an error.\n"
            "This question asks for RA, TRC, DRC expressions for 11 queries. "
            "Queries 8, 10, 11 cannot be expressed in RA/RC — explain why. "
            "Evaluate correctness of relational algebra notation and SQL equivalents. "
            "Key: Boeing aircraft check uses aname LIKE 'Boeing%%' or = 'Boeing'."
        ),
    },

    # ID: 09f1cb01bac5 — Unsafe query definition (Exercise 4.7) — pure conceptual
    "09f1cb01bac5": {
        "judge_hint": (
            "Conceptual question about unsafe queries in relational calculus. "
            "No DB execution needed. "
            "An unsafe query is one that returns an infinite number of tuples "
            "(e.g., all things NOT in a relation). "
            "Key example: {S | ¬(S ∈ Sailors)} — infinite result. "
            "Important because we cannot return complete infinite answer sets."
        ),
    },

    # ID: f10e4b184023 — Implementing outer joins (Exercise 14.7) — algorithm/conceptual
    "f10e4b184023": {
        "schema_fixture": _SAILORS_DDL + _RESERVES_DDL,
        "expected_rows": [],
        "judge_hint": (
            "Schema: Sailors(sid,sname,rating,age), Reserves(sid,bid,day). "
            "Question asks how to modify 4 join algorithms (BNLJ, INLJ, SMJ, Hash Join) "
            "to directly compute LEFT/RIGHT/FULL OUTER JOINs without a post-join comparison. "
            "Key technique for each: maintain bit flags for unmatched tuples, "
            "output them with NULL values for the other relation's attributes. "
            "Evaluate correctness of the algorithmic descriptions for all 3 outer join variants."
        ),
    },

    # ID: 316cc7c034d5 — CC protocol Venn diagram (Exercise 17.7) — conceptual/diagram
    "316cc7c034d5": {
        "judge_hint": (
            "Conceptual question asking for a Venn diagram of concurrency control protocol "
            "schedule classes. No DB execution needed. "
            "Hierarchy (innermost → outermost): "
            "Conservative 2PL ⊂ Strict 2PL ⊂ 2PL ⊂ Conflict-Serializable ⊂ Serializable. "
            "Timestamp w/o TWR and Timestamp with TWR partially overlap 2PL. "
            "Multiversion schedules overlap with Serializable but not necessarily Conflict-Serializable. "
            "Optimistic CC allows schedules outside Conflict-Serializable. "
            "Evaluate the correctness of the described hierarchy and example schedules."
        ),
    },

    # ID: 2e8fe7d4d629 — Data Warehousing and OLAP (Exercise 25.1) — conceptual
    "2e8fe7d4d629": {
        "judge_hint": (
            "Conceptual question about data warehousing and OLAP. No DB execution needed. "
            "Key points to evaluate: "
            "1) Warehousing=historical integration, OLAP=analysis, mining=patterns — complementary. "
            "2) Warehousing uses asynchronous replication (avoids blocking OLTP). "
            "3) Metadata repository: ETL rules, lineage, business definitions — broader than catalog. "
            "4) Star schema: central fact table + denormalized dimensions (NOT BCNF). "
            "5) MOLAP=multidimensional arrays, ROLAP=relational tables. "
            "6) Data mining discovers hidden patterns; OLAP is analyst-driven slicing/dicing."
        ),
    },

    # =========================================================================
    # ADDITIONAL TOPICS
    # =========================================================================

    "30541b0a864e": {
        "schema_fixture": _SALES_TIMES_DDL,
        "expected_rows": [
            {"year": 2020, "year_total": 250000.0},
            {"year": 2021, "year_total": 300000.0},
            {"year": 2022, "year_total": 200000.0},
        ],
        "judge_hint": (
            "Schema: Times(timeid,year,month,day), Sales(saleid,timeid,sales). "
            "Question asks about WINDOW clause vs GROUP BY, window frames, and "
            "whether a GROUP BY query can be rewritten with WINDOW. "
            "The GROUP BY query for year totals CAN be rewritten as: "
            "SELECT DISTINCT T.year, SUM(S.sales) OVER (PARTITION BY T.year) "
            "FROM Sales S JOIN Times T ON S.timeid = T.timeid. "
            "Running total example uses ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW."
        ),
    },

    "9d080a66a9fc": {
        "schema_fixture": _NUM_RESERVATIONS_DDL,
        "expected_rows": [
            {"sid": 22, "sname": "ahab", "numres": 4},
        ],
        "judge_hint": (
            "View NumReservations(sid,sname,numres) counts reservations per sailor. "
            "Data: ahab(22) has 4 reservations (max), jonah(41) has 0 (not in reserves). "
            "Query modification of MAX(numres): substitute view definition inline and "
            "use WHERE numres = (SELECT MAX(numres) FROM <view>). "
            "Materialization pros/cons: materialized = fast but stale; "
            "on-demand = always fresh but slow for large tables."
        ),
    },

    # ID: f4ddd3bdb9b0 — WR/RW/WW conflict schedules + Strict 2PL (Exercise 16.3)
    # No SQL harness (no schema_fixture / expected_rows) — judge_hint only.
    "f4ddd3bdb9b0": {
        "judge_hint": (
            "EVALUATION INSTRUCTIONS — READ BEFORE SCORING:\n"
            "This question asks for EXAMPLE schedules that demonstrate write-read (WR), "
            "read-write (RW), and write-write (WW) conflicts, plus an explanation of how "
            "Strict 2PL disallows each.\n"
            "CRITICAL: The question says 'Give AN EXAMPLE schedule' — it does NOT require "
            "the specific schedules from the textbook answer. Any schedule that correctly "
            "demonstrates the stated conflict type is FULLY CORRECT.\n"
            "DO NOT penalise or flag as a hallucination any schedule that differs from the "
            "expected answer's specific sequences, provided it correctly illustrates the conflict.\n"
            "Criteria for a correct WR example: T_i writes X, then T_j reads X, before T_i commits (dirty read).\n"
            "Criteria for a correct RW example: T_i reads X, then T_j writes X, before T_i commits (unrepeatable read).\n"
            "Criteria for a correct WW example: T_i writes X, then T_j writes X, before either commits (overwrite of uncommitted data).\n"
            "Strict 2PL disallows all three by holding all locks until commit — any correct "
            "explanation of this mechanism is acceptable regardless of phrasing."
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers for the loader to use
# ---------------------------------------------------------------------------

def get_fixture(question_id: str) -> dict:
    """
    Return the fixture dict for a question, or an empty dict if none defined.
    The 12-char hex ID from the question bank JSON is used directly as the key.
    """
    return SQL_FIXTURES.get(question_id, {})


def get_schema_fixture(question_id: str) -> str | None:
    """Return the DDL schema fixture for a question, or None."""
    return SQL_FIXTURES.get(question_id, {}).get("schema_fixture")


def get_expected_rows(question_id: str) -> list | None:
    """Return the expected result rows for a question, or None."""
    return SQL_FIXTURES.get(question_id, {}).get("expected_rows")


def get_judge_hint(question_id: str) -> str | None:
    """Return the judge hint string for a question, or None."""
    return SQL_FIXTURES.get(question_id, {}).get("judge_hint")


def get_explain_context(question_id: str) -> str | None:
    """Return EXPLAIN ANALYZE output for query-optimisation questions, or None."""
    return SQL_FIXTURES.get(question_id, {}).get("explain_context")


def list_questions_with_fixtures() -> list[str]:
    """Return list of question IDs that have at least a schema_fixture defined."""
    return [
        qid for qid, f in SQL_FIXTURES.items()
        if f.get("schema_fixture")
    ]


# ---------------------------------------------------------------------------
# Per-question reverse-translation maps:  fixture_name  →  natural_name
#
# Used by eval_service._apply_rename_map() to translate the model answer from
# fixture-specific identifiers back to natural names BEFORE the Judge LLM sees
# the answer.  The SQL harness uses fixture names for execution; the Judge sees
# natural names that match its training data, eliminating false-positive
# hallucination reports about "syntax errors" on valid renamed tables.
# ---------------------------------------------------------------------------
_FIXTURE_RENAME_MAPS: dict[str, dict[str, str]] = {
    # Flights/Aircraft/Certified/Employees — SQL queries (Exercise 5.3)
    "0def7bdc534e": {
        "femployees": "Employees",
        "ffrom":      "from",
        "fto":        "to",
    },
    # Flights/Aircraft/Certified/Employees — RA/TRC/DRC (Exercise 4.5)
    "8c61735b507d": {
        "femployees": "Employees",
        "ffrom":      "from",
        "fto":        "to",
    },
    # 7f30162b5f47: ternary DDL question — model creates its own tables, no rename needed
}


def get_fixture_rename_map(question_id: str) -> dict[str, str]:
    """
    Return the reverse-translation map for a question.

    Maps fixture-specific identifiers (e.g. femployees, ffrom, fto) to the
    natural names used in the question text and known to the Judge LLM
    (e.g. Employees, from, to).

    Returns an empty dict if no translation is needed for this question.
    """
    return _FIXTURE_RENAME_MAPS.get(question_id, {})


# ---------------------------------------------------------------------------
# Tag-based routing helpers
# ---------------------------------------------------------------------------
# After the question bank correction, question_type is ONLY 'conceptual' or
# 'practical'. Content-specific routing (SQL harness, rubric selection, format
# compliance) must now be driven by the `tags` list on each question.
#
# Use these helpers everywhere instead of comparing question.question_type to
# legacy values like 'sql', 'schema', 'transaction', 'query_optimization'.
# ---------------------------------------------------------------------------

def needs_sql_harness(question_type: str, tags: list[str]) -> bool:
    """
    True if the SQL execution harness should run for this question.
    A question needs the harness when it is practical AND tagged 'sql'
    (i.e., the challenger model is expected to produce executable SQL).
    """
    return question_type == "practical" and "sql" in tags


def needs_sql_rubric(tags: list[str]) -> bool:
    """
    True if the Judge should use the SQL-specific scoring rubric.
    Applied to ALL sql-tagged questions (conceptual or practical) since
    even conceptual sql questions discuss SQL constructs.
    """
    return "sql" in tags


def needs_ddl_format_check(question_type: str, tags: list[str]) -> bool:
    """
    True if the format compliance checker should verify DDL/schema formatting.
    Applied to practical questions tagged 'sql' (they produce CREATE TABLE etc.)
    """
    return question_type == "practical" and "sql" in tags


def needs_numbered_steps_check(question_type: str, tags: list[str]) -> bool:
    """
    True if the format compliance checker should verify numbered-step formatting.
    Applied to practical questions tagged 'transactions' (schedule traces,
    lock sequences, ARIES log traces produce enumerated step-by-step output).
    """
    return question_type == "practical" and "transactions" in tags


def get_db_correctness_route(question_type: str, tags: list[str]) -> str:
    """
    Return the scoring route key for the DB correctness pillar.
    Maps (question_type, tags) to one of the keys in DBCorrectnessBundle.

    Routes:
      practical + sql       → 'sql'          (SQL execution scores dominate)
      practical + trans     → 'transaction'   (schedule/lock/ARIES traces)
      conceptual + trans    → 'transaction'
      conceptual + normal   → 'normalization' (maps to conceptual scorer)
      practical + no-sql    → 'practical'     (maps to conceptual scorer)
      everything else       → 'conceptual'
    """
    if "sql" in tags and question_type == "practical":
        return "sql"
    if "transactions" in tags:
        return "transaction"
    if "normalization" in tags:
        return "normalization"
    return "conceptual"
