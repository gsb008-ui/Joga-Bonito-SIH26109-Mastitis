import sqlite3

DATABASE_NAME = "mastitis.db"


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_tag TEXT UNIQUE NOT NULL,
            breed TEXT,
            age INTEGER,
            lactation_num INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_tag TEXT NOT NULL,
            record_date TEXT NOT NULL,
            days_since_calving INTEGER,
            prior_mastitis INTEGER,
            milk_yield REAL,
            body_temp REAL,
            milk_conductivity REAL,
            activity_score REAL,

            FOREIGN KEY (cow_tag) REFERENCES cows(cow_tag)
        )
    """)
    cursor.execute("""
    INSERT OR IGNORE INTO cows (cow_tag, breed, age, lactation_num)
    VALUES ('A127', 'Jersey', 5, 3)
""")
    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_database()
    print("Database created successfully!")