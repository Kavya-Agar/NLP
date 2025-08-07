import pandas as pd

df = pd.read_csv('datasets/BCG Financial Data.csv')

def get_total_revenue(company, year):
    result = df[(df['Company'].str.lower() == company.lower()) & (df['Fiscal Year'] == int(year))]
    if not result.empty:
        return f"{company}'s total revenue in {year} is ${float(result['Total Revenue'].values[0]):.2e}"
    return "Data not found."

def get_net_income_change(company, year1, year2):
    data1 = df[(df['Company'].str.lower() == company.lower()) & (df['Fiscal Year'] == int(year1))]
    data2 = df[(df['Company'].str.lower() == company.lower()) & (df['Fiscal Year'] == int(year2))]
    if not data1.empty and not data2.empty:
        income1 = float(data1['Net Income'].values[0])
        income2 = float(data2['Net Income'].values[0])
        diff = income2 - income1
        direction = "increased" if diff >= 0 else "decreased"
        return f"{company}'s net income has {direction} by ${abs(diff):.2e} from {year1} to {year2}."
    return "Data not found."

def get_total_assets(company, year):
    result = df[(df['Company'].str.lower() == company.lower()) & (df['Fiscal Year'] == int(year))]
    if not result.empty:
        return f"{company}'s total assets in {year} is ${float(result['Total Assets'].values[0]):.2e}"
    return "Data not found."

def get_cash_flow(company, year):
    result = df[(df['Company'].str.lower() == company.lower()) & (df['Fiscal Year'] == int(year))]
    if not result.empty:
        return f"{company}'s cash flow from operating activities in {year} is ${float(result['Cash Flow from Operating Activities'].values[0]):.2e}"
    return "Data not found."

def get_liabilities(company, year):
    result = df[(df['Company'].str.lower() == company.lower()) & (df['Fiscal Year'] == int(year))]
    if not result.empty:
        return f"{company}'s total liabilities in {year} is ${float(result['Total Liabilities'].values[0]):.2e}"
    return "Data not found."

def simple_chatbot():
    print("Welcome to the Financial Chatbot Prototype!")
    print("Try questions like:")
    print("- What is the total revenue for Apple in 2024?")
    print("- How has net income changed for Microsoft from 2022 to 2023?")
    print("- What are the total assets for Tesla in 2023?")
    print("- What is the cash flow from operating activities for Apple in 2023?")
    print("- What are the total liabilities for Microsoft in 2024?")
    print("Type 'exit' to end.\n")

    while True:
        user_input = input("Ask your question: ").lower()
        if 'exit' in user_input:
            print("Goodbye!")
            break

        found = False

        if "total revenue" in user_input:
            for company in df['Company'].unique():
                if company.lower() in user_input:
                    for year in df['Fiscal Year'].unique():
                        if str(year) in user_input:
                            print(get_total_revenue(company, year))
                            found = True
                            break
            if not found:
                print("Please specify the company and year for total revenue.")

        elif "net income" in user_input and "change" in user_input:
            for company in df['Company'].unique():
                if company.lower() in user_input:
                    years = [int(word) for word in user_input.split() if word.isdigit()]
                    if len(years) == 2:
                        print(get_net_income_change(company, years[0], years[1]))
                        found = True
                        break
            if not found:
                print("Please specify the company and two years for net income change (e.g., from 2022 to 2023).")

        elif "total assets" in user_input:
            for company in df['Company'].unique():
                if company.lower() in user_input:
                    for year in df['Fiscal Year'].unique():
                        if str(year) in user_input:
                            print(get_total_assets(company, year))
                            found = True
                            break
            if not found:
                print("Please specify the company and year for total assets.")

        elif "cash flow" in user_input:
            for company in df['Company'].unique():
                if company.lower() in user_input:
                    for year in df['Fiscal Year'].unique():
                        if str(year) in user_input:
                            print(get_cash_flow(company, year))
                            found = True
                            break
            if not found:
                print("Please specify the company and year for cash flow from operating activities.")

        elif "total liabilities" in user_input:
            for company in df['Company'].unique():
                if company.lower() in user_input:
                    for year in df['Fiscal Year'].unique():
                        if str(year) in user_input:
                            print(get_liabilities(company, year))
                            found = True
                            break
            if not found:
                print("Please specify the company and year for total liabilities.")

        else:
            print("Sorry, I can only answer predefined queries about total revenue, net income change, total assets, cash flow from operating activities, or total liabilities. Please try again!")

if __name__ == "__main__":
    simple_chatbot()
