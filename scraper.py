import csv
import os
import pandas as pd
from jobspy import scrape_jobs

def truncate_job_listings(csv_path, max_jobs=5000, remove_count=1000):
    if os.path.exists(csv_path):
        jobs = pd.read_csv(csv_path)
        if len(jobs) > max_jobs:
            print(f"Truncating job listings to the last {max_jobs - remove_count} jobs...")
            jobs = jobs.iloc[-(max_jobs - remove_count):]  # Keep the newest jobs
            jobs.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\")
            print(f"Truncated job listings. Remaining jobs: {len(jobs)}.")

def main(search_term="", location = "United States"):
    output_file = "job_listings.csv"

    # Define the job platforms to scrape
    platforms = ["indeed"]

    # Scrape job listings
    jobs = scrape_jobs(
        site_name=platforms,
        search_term=search_term,  # No fixed job role
        location=location,
        results_wanted=1000,  # Number of results
        hours_old=336,       # Jobs posted in the last 2 weeks
        country_indeed='USA',  # Country code for Indeed
    )

    # Drop unnecessary columns
    jobs = jobs.drop(['id','site', 'salary_source','interval','job_level', 'job_function', 'listing_type', 'emails','company_industry','company_url', 'company_logo', 'company_url_direct', 'company_addresses', 'company_num_employees', 'company_revenue', 'company_description'], axis=1)

    # Check if the output file exists
    if os.path.exists(output_file):
        # Read the existing file
        existing_jobs = pd.read_csv(output_file)

        # Concatenate the new jobs with existing jobs
        all_jobs = pd.concat([existing_jobs, jobs], ignore_index=True)

        # Drop duplicate entries
        all_jobs = all_jobs.drop_duplicates()

        # Save the updated job listings back to the file
        all_jobs.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\")
        print(f"Appended {len(jobs)} jobs to '{output_file}', avoiding duplicates. Total jobs: {len(all_jobs)}.")

        # Truncate the job listings if too large
        truncate_job_listings(output_file)
    else:
        # Save the new job listings to the file
        jobs.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\")
        print(f"Saved {len(jobs)} jobs to '{output_file}'.")

if __name__ == "__main__":
    main()
