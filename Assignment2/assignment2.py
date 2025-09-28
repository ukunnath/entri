# student_management_system.py

import csv
import datetime
import os
import getpass

# ------------------ CONFIG ------------------
DATA_FILE = "students.csv"
LOGIN_CREDENTIALS = {"admin": "admin123", "teacher": "teach123"}

# ------------------ UTIL MODULE ------------------
def validate_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter an integer.")

def validate_non_empty(prompt):
    while True:
        data = input(prompt).strip()
        if data:
            return data
        print("Input cannot be empty.")

def generate_student_id():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"SID{timestamp}"

def print_student(student):
    print(f"ID: {student['ID']}, Name: {student['Name']}, Roll: {student['Roll']}, "
          f"Age: {student['Age']}, Dept: {student['Department']}, "
          f"Marks: {student['Marks']}, Entry Time: {student['EntryTime']}")

# ------------------ FILE HANDLING ------------------
def load_students():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        return list(reader)

def save_students(students):
    with open(DATA_FILE, mode='w', newline='') as file:
        fieldnames = ['ID', 'Name', 'Roll', 'Age', 'Department', 'Marks', 'EntryTime']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(students)

# ------------------ LOGIN SYSTEM ------------------
def login():
    print("\n=== Login ===")
    username = input("Username: ").strip()
    password = input("password: ").strip()
    #password = getpass.getpass("Password: ")
    if username in LOGIN_CREDENTIALS and LOGIN_CREDENTIALS[username] == password:
        print("Login successful!\n")
        return True
    else:
        print("Invalid credentials.\n")
        return False

# ------------------ MAIN OPERATIONS ------------------
def add_student(students):
    print("\n--- Add New Student ---")
    student = {
        "ID": generate_student_id(),
        "Name": validate_non_empty("Enter Name: "),
        "Roll": validate_non_empty("Enter Roll Number: "),
        "Age": str(validate_int("Enter Age: ")),
        "Department": validate_non_empty("Enter Department: "),
        "Marks": str(validate_int("Enter Marks: ")),
        "EntryTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    students.append(student)
    save_students(students)
    print("Student added successfully.\n")

def view_students(students):
    print("\n--- Student Records ---")
    if not students:
        print("No student records found.\n")
        return
    for student in students:
        print_student(student)
    print()

def update_student(students):
    print("\n--- Update Student ---")
    roll = input("Enter Roll Number to update: ").strip()
    for student in students:
        if student['Roll'] == roll:
            print("Enter new details (leave blank to keep current value):")
            student['Name'] = input(f"Name [{student['Name']}]: ") or student['Name']
            student['Age'] = input(f"Age [{student['Age']}]: ") or student['Age']
            student['Department'] = input(f"Department [{student['Department']}]: ") or student['Department']
            new_marks = input(f"Marks [{student['Marks']}]: ")
            if new_marks:
                student['Marks'] = new_marks
            save_students(students)
            print("Student record updated.\n")
            return
    print("Student with given Roll Number not found.\n")

def delete_student(students):
    print("\n--- Delete Student ---")
    roll = input("Enter Roll Number to delete: ").strip()
    for student in students:
        if student['Roll'] == roll:
            students.remove(student)
            save_students(students)
            print("Student record deleted.\n")
            return
    print("Student not found.\n")

def search_student(students):
    print("\n--- Search Student ---")
    key = input("Search by Name or Department: ").strip().lower()
    found = [s for s in students if key in s['Name'].lower() or key in s['Department'].lower()]
    if found:
        for s in found:
            print_student(s)
    else:
        print("No matching records found.\n")

def top_students(students):
    print("\n--- Top Performing Students ---")
    if not students:
        print("No records to analyze.\n")
        return
    students_sorted = sorted(students, key=lambda x: int(x['Marks']), reverse=True)
    top = students_sorted[:3]
    for s in top:
        print_student(s)

# ------------------ MAIN LOOP ------------------
def main():
    if not login():
        return
    students = load_students()

    while True:
        print("======= Student Management System =======")
        print("1. Add Student")
        print("2. View Students")
        print("3. Update Student")
        print("4. Delete Student")
        print("5. Search Student")
        print("6. Top Performing Students")
        print("7. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            add_student(students)
        elif choice == "2":
            view_students(students)
        elif choice == "3":
            update_student(students)
        elif choice == "4":
            delete_student(students)
        elif choice == "5":
            search_student(students)
        elif choice == "6":
            top_students(students)
        elif choice == "7":
            print("Exiting system. Goodbye!\n")
            break
        else:
            print("Invalid choice. Please try again.\n")

if __name__ == "__main__":
    main()
