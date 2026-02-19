"""
Industry EV/EBITDA Multiples from Damodaran (January 2026)
==========================================================
Source: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/vebitda.html

These multiples are used for terminal value calculation in DCF analysis
using the exit multiple method.
"""

# Damodaran Industry Multiples (January 2026)
# Key: Industry Name (as per Damodaran classification)
# Value: EV/EBITDA multiple (None = Not Applicable, e.g., banks)
DAMODARAN_EV_EBITDA_MULTIPLES = {
    "Advertising": 12.0,
    "Aerospace/Defense": 21.58,
    "Air Transport": 7.58,
    "Apparel": 10.3,
    "Auto & Truck": 47.76,
    "Auto Parts": 6.43,
    "Bank (Money Center)": None,
    "Banks (Regional)": None,
    "Beverage (Alcoholic)": 8.61,
    "Beverage (Soft)": 16.9,
    "Broadcasting": 7.85,
    "Brokerage & Investment Banking": None,
    "Building Materials": 11.61,
    "Business & Consumer Services": 14.26,
    "Cable TV": 6.21,
    "Chemical (Basic)": 8.57,
    "Chemical (Diversified)": 8.39,
    "Chemical (Specialty)": 13.36,
    "Coal & Related Energy": 10.37,
    "Computer Services": 14.1,
    "Computers/Peripherals": 25.42,
    "Construction Supplies": 16.82,
    "Diversified": 11.42,
    "Drugs (Biotechnology)": 15.78,
    "Drugs (Pharmaceutical)": 15.25,
    "Education": 9.26,
    "Electrical Equipment": 24.59,
    "Electronics (Consumer & Office)": 30.7,
    "Electronics (General)": 19.99,
    "Engineering/Construction": 17.18,
    "Entertainment": 19.41,
    "Environmental & Waste Services": 15.61,
    "Farming/Agriculture": 16.04,
    "Financial Svcs. (Non-bank & Insurance)": 57.52,
    "Food Processing": 10.01,
    "Food Wholesalers": 11.08,
    "Furn/Home Furnishings": 11.27,
    "Green & Renewable Energy": 13.44,
    "Healthcare Products": 19.78,
    "Healthcare Support Services": 11.17,
    "Heathcare Information and Technology": 21.27,
    "Homebuilding": 8.92,
    "Hospitals/Healthcare Facilities": 8.86,
    "Hotel/Gaming": 14.93,
    "Household Products": 13.17,
    "Information Services": 11.5,
    "Insurance (General)": 15.76,
    "Insurance (Life)": 12.52,
    "Insurance (Prop/Cas.)": 8.44,
    "Investments & Asset Management": 38.03,
    "Machinery": 16.22,
    "Metals & Mining": 11.39,
    "Office Equipment & Services": 8.59,
    "Oil/Gas (Integrated)": 8.16,
    "Oil/Gas (Production and Exploration)": 5.15,
    "Oil/Gas Distribution": 11.56,
    "Oilfield Svcs/Equip.": 8.63,
    "Packaging & Container": 9.71,
    "Paper/Forest Products": 8.18,
    "Power": 12.38,
    "Precious Metals": 10.68,
    "Publishing & Newspapers": 11.24,
    "R.E.I.T.": 19.87,
    "Real Estate (Development)": 10.23,
    "Real Estate (General/Diversified)": 17.29,
    "Real Estate (Operations & Services)": 21.95,
    "Recreation": 10.39,
    "Reinsurance": 8.67,
    "Restaurant/Dining": 17.49,
    "Retail (Automotive)": 14.79,
    "Retail (Building Supply)": 14.42,
    "Retail (Distributors)": 13.71,
    "Retail (General)": 17.38,
    "Retail (Grocery and Food)": 8.94,
    "Retail (REITs)": 16.73,
    "Retail (Special Lines)": 11.47,
    "Rubber& Tires": 6.74,
    "Semiconductor": 34.75,
    "Semiconductor Equip": 24.74,
    "Shipbuilding & Marine": 7.95,
    "Shoe": 16.86,
    "Software (Entertainment)": 22.01,
    "Software (Internet)": 30.26,
    "Software (System & Application)": 24.48,
    "Steel": 11.59,
    "Telecom (Wireless)": 8.97,
    "Telecom. Equipment": 24.07,
    "Telecom. Services": 6.54,
    "Tobacco": 13.46,
    "Transportation": 12.55,
    "Transportation (Railroads)": 13.49,
    "Trucking": 10.41,
    "Utility (General)": 13.73,
    "Utility (Water)": 14.14,
}

# Industry mapping from yfinance sector/industry to Damodaran classification
# yfinance provides: sector (broad) and industry (specific)
YFINANCE_TO_DAMODARAN_MAPPING = {
    # Technology sector
    "Consumer Electronics": "Electronics (Consumer & Office)",
    "Software—Infrastructure": "Software (System & Application)",
    "Software—Application": "Software (System & Application)",
    "Internet Content & Information": "Software (Internet)",
    "Internet Retail": "Software (Internet)",
    "Semiconductors": "Semiconductor",
    "Semiconductor Equipment & Materials": "Semiconductor Equip",
    "Computer Hardware": "Computers/Peripherals",
    "Information Technology Services": "Computer Services",
    "Electronic Components": "Electronics (General)",
    "Scientific & Technical Instruments": "Electronics (General)",
    "Communication Equipment": "Telecom. Equipment",
    "Software - Infrastructure": "Software (System & Application)",
    "Software - Application": "Software (System & Application)",
    
    # Healthcare sector
    "Drug Manufacturers—General": "Drugs (Pharmaceutical)",
    "Drug Manufacturers—Specialty & Generic": "Drugs (Pharmaceutical)",
    "Biotechnology": "Drugs (Biotechnology)",
    "Medical Devices": "Healthcare Products",
    "Medical Instruments & Supplies": "Healthcare Products",
    "Health Information Services": "Heathcare Information and Technology",
    "Healthcare Plans": "Healthcare Support Services",
    "Medical Care Facilities": "Hospitals/Healthcare Facilities",
    "Diagnostics & Research": "Drugs (Biotechnology)",
    "Pharmaceutical Retailers": "Retail (Special Lines)",
    
    # Financial sector
    "Banks—Diversified": "Bank (Money Center)",
    "Banks—Regional": "Banks (Regional)",
    "Asset Management": "Investments & Asset Management",
    "Insurance—Diversified": "Insurance (General)",
    "Insurance—Life": "Insurance (Life)",
    "Insurance—Property & Casualty": "Insurance (Prop/Cas.)",
    "Insurance—Reinsurance": "Reinsurance",
    "Capital Markets": "Brokerage & Investment Banking",
    "Financial Data & Stock Exchanges": "Financial Svcs. (Non-bank & Insurance)",
    "Credit Services": "Financial Svcs. (Non-bank & Insurance)",
    
    # Consumer Cyclical
    "Auto Manufacturers": "Auto & Truck",
    "Auto Parts": "Auto Parts",
    "Restaurants": "Restaurant/Dining",
    "Apparel Manufacturing": "Apparel",
    "Apparel Retail": "Apparel",
    "Footwear & Accessories": "Shoe",
    "Specialty Retail": "Retail (Special Lines)",
    "Home Improvement Retail": "Retail (Building Supply)",
    "Department Stores": "Retail (General)",
    "Discount Stores": "Retail (General)",
    "Luxury Goods": "Apparel",
    "Residential Construction": "Homebuilding",
    "Furnishings, Fixtures & Appliances": "Furn/Home Furnishings",
    "Leisure": "Recreation",
    "Gambling": "Hotel/Gaming",
    "Resorts & Casinos": "Hotel/Gaming",
    "Lodging": "Hotel/Gaming",
    "Travel Services": "Recreation",
    "Entertainment": "Entertainment",
    "Broadcasting": "Broadcasting",
    
    # Consumer Defensive
    "Beverages—Non-Alcoholic": "Beverage (Soft)",
    "Beverages—Brewers": "Beverage (Alcoholic)",
    "Beverages—Wineries & Distilleries": "Beverage (Alcoholic)",
    "Packaged Foods": "Food Processing",
    "Food Distribution": "Food Wholesalers",
    "Grocery Stores": "Retail (Grocery and Food)",
    "Household & Personal Products": "Household Products",
    "Tobacco": "Tobacco",
    "Education & Training Services": "Education",
    "Farm Products": "Farming/Agriculture",
    "Agricultural Inputs": "Farming/Agriculture",
    
    # Industrials
    "Aerospace & Defense": "Aerospace/Defense",
    "Airlines": "Air Transport",
    "Railroads": "Transportation (Railroads)",
    "Trucking": "Trucking",
    "Integrated Freight & Logistics": "Transportation",
    "Marine Shipping": "Shipbuilding & Marine",
    "Industrial Distribution": "Retail (Distributors)",
    "Specialty Industrial Machinery": "Machinery",
    "Farm & Heavy Construction Machinery": "Machinery",
    "Building Products & Equipment": "Building Materials",
    "Engineering & Construction": "Engineering/Construction",
    "Waste Management": "Environmental & Waste Services",
    "Consulting Services": "Business & Consumer Services",
    "Staffing & Employment Services": "Business & Consumer Services",
    "Security & Protection Services": "Business & Consumer Services",
    "Rental & Leasing Services": "Business & Consumer Services",
    "Electrical Equipment & Parts": "Electrical Equipment",
    "Conglomerates": "Diversified",
    "Metal Fabrication": "Steel",
    "Tools & Accessories": "Machinery",
    "Pollution & Treatment Controls": "Environmental & Waste Services",
    
    # Energy
    "Oil & Gas Integrated": "Oil/Gas (Integrated)",
    "Oil & Gas E&P": "Oil/Gas (Production and Exploration)",
    "Oil & Gas Midstream": "Oil/Gas Distribution",
    "Oil & Gas Equipment & Services": "Oilfield Svcs/Equip.",
    "Oil & Gas Refining & Marketing": "Oil/Gas (Integrated)",
    "Thermal Coal": "Coal & Related Energy",
    "Uranium": "Power",
    
    # Basic Materials
    "Specialty Chemicals": "Chemical (Specialty)",
    "Chemicals": "Chemical (Basic)",
    "Agricultural Inputs": "Chemical (Specialty)",
    "Aluminum": "Metals & Mining",
    "Copper": "Metals & Mining",
    "Steel": "Steel",
    "Gold": "Precious Metals",
    "Silver": "Precious Metals",
    "Other Precious Metals & Mining": "Precious Metals",
    "Other Industrial Metals & Mining": "Metals & Mining",
    "Lumber & Wood Production": "Paper/Forest Products",
    "Paper & Paper Products": "Paper/Forest Products",
    "Packaging & Containers": "Packaging & Container",
    "Building Materials": "Building Materials",
    "Coking Coal": "Coal & Related Energy",
    
    # Real Estate
    "REIT—Diversified": "R.E.I.T.",
    "REIT—Industrial": "R.E.I.T.",
    "REIT—Office": "R.E.I.T.",
    "REIT—Retail": "Retail (REITs)",
    "REIT—Residential": "R.E.I.T.",
    "REIT—Healthcare Facilities": "R.E.I.T.",
    "REIT—Hotel & Motel": "Hotel/Gaming",
    "REIT—Specialty": "R.E.I.T.",
    "Real Estate—Development": "Real Estate (Development)",
    "Real Estate—Diversified": "Real Estate (General/Diversified)",
    "Real Estate Services": "Real Estate (Operations & Services)",
    
    # Utilities
    "Utilities—Regulated Electric": "Utility (General)",
    "Utilities—Regulated Gas": "Utility (General)",
    "Utilities—Diversified": "Utility (General)",
    "Utilities—Regulated Water": "Utility (Water)",
    "Utilities—Renewable": "Green & Renewable Energy",
    "Utilities—Independent Power Producers": "Power",
    
    # Communication Services
    "Telecom Services": "Telecom. Services",
    "Advertising Agencies": "Advertising",
    "Publishing": "Publishing & Newspapers",
    "Electronic Gaming & Multimedia": "Software (Entertainment)",
    
    # Additional mappings for common yfinance industries
    "Internet Software & Services": "Software (Internet)",
    "Data Processing & Outsourced Services": "Computer Services",
    "Application Software": "Software (System & Application)",
    "Systems Software": "Software (System & Application)",
    "Pharmaceuticals": "Drugs (Pharmaceutical)",
    "Health Care Equipment": "Healthcare Products",
    "Health Care Services": "Healthcare Support Services",
    "Diversified Banks": "Bank (Money Center)",
    "Regional Banks": "Banks (Regional)",
    "Investment Banking & Brokerage": "Brokerage & Investment Banking",
    "Automobile Manufacturers": "Auto & Truck",
    "Automotive Retail": "Retail (Automotive)",
}

# Sector-level fallback mapping (if specific industry not found)
SECTOR_FALLBACK_MAPPING = {
    "Technology": "Software (System & Application)",
    "Healthcare": "Healthcare Products",
    "Financial Services": "Financial Svcs. (Non-bank & Insurance)",
    "Consumer Cyclical": "Retail (General)",
    "Consumer Defensive": "Food Processing",
    "Industrials": "Machinery",
    "Energy": "Oil/Gas (Integrated)",
    "Basic Materials": "Chemical (Basic)",
    "Real Estate": "R.E.I.T.",
    "Utilities": "Utility (General)",
    "Communication Services": "Entertainment",
}


def get_industry_multiple(yf_industry: str, yf_sector: str = None) -> tuple:
    """
    Get the Damodaran EV/EBITDA multiple for a company based on its industry.
    
    Args:
        yf_industry: Industry string from yfinance (e.g., "Software—Application")
        yf_sector: Sector string from yfinance as fallback (e.g., "Technology")
    
    Returns:
        tuple: (multiple, damodaran_industry_name, is_exact_match)
        - multiple: float or None if industry uses different valuation (e.g., banks)
        - damodaran_industry_name: The matched Damodaran industry
        - is_exact_match: True if direct mapping found, False if sector fallback used
    """
    if not yf_industry:
        # No industry provided, use sector fallback
        if yf_sector and yf_sector in SECTOR_FALLBACK_MAPPING:
            damodaran_industry = SECTOR_FALLBACK_MAPPING[yf_sector]
            return (
                DAMODARAN_EV_EBITDA_MULTIPLES.get(damodaran_industry),
                damodaran_industry,
                False
            )
        return (None, None, False)
    
    # Clean up industry string (handle various dash types)
    clean_industry = yf_industry.replace("—", "-").replace("–", "-")
    
    # Try exact match first
    if clean_industry in YFINANCE_TO_DAMODARAN_MAPPING:
        damodaran_industry = YFINANCE_TO_DAMODARAN_MAPPING[clean_industry]
        return (
            DAMODARAN_EV_EBITDA_MULTIPLES.get(damodaran_industry),
            damodaran_industry,
            True
        )
    
    # Try with original string
    if yf_industry in YFINANCE_TO_DAMODARAN_MAPPING:
        damodaran_industry = YFINANCE_TO_DAMODARAN_MAPPING[yf_industry]
        return (
            DAMODARAN_EV_EBITDA_MULTIPLES.get(damodaran_industry),
            damodaran_industry,
            True
        )
    
    # Try partial matching for common keywords
    industry_lower = yf_industry.lower()
    
    partial_matches = {
        "software": "Software (System & Application)",
        "semiconductor": "Semiconductor",
        "internet": "Software (Internet)",
        "pharma": "Drugs (Pharmaceutical)",
        "biotech": "Drugs (Biotechnology)",
        "bank": "Banks (Regional)",
        "insurance": "Insurance (General)",
        "retail": "Retail (General)",
        "restaurant": "Restaurant/Dining",
        "aerospace": "Aerospace/Defense",
        "auto": "Auto & Truck",
        "oil": "Oil/Gas (Integrated)",
        "utility": "Utility (General)",
        "telecom": "Telecom. Services",
        "healthcare": "Healthcare Products",
        "chemical": "Chemical (Basic)",
        "media": "Entertainment",
        "entertainment": "Entertainment",
        "real estate": "Real Estate (General/Diversified)",
        "reit": "R.E.I.T.",
        "food": "Food Processing",
        "beverage": "Beverage (Soft)",
    }
    
    for keyword, damodaran_industry in partial_matches.items():
        if keyword in industry_lower:
            return (
                DAMODARAN_EV_EBITDA_MULTIPLES.get(damodaran_industry),
                damodaran_industry,
                False  # Partial match, not exact
            )
    
    # Fall back to sector mapping
    if yf_sector and yf_sector in SECTOR_FALLBACK_MAPPING:
        damodaran_industry = SECTOR_FALLBACK_MAPPING[yf_sector]
        return (
            DAMODARAN_EV_EBITDA_MULTIPLES.get(damodaran_industry),
            damodaran_industry,
            False
        )
    
    return (None, None, False)


def get_all_multiples_table() -> list:
    """
    Get the full Damodaran multiples table as a list of dicts for display.
    
    Returns:
        List of dicts with 'industry' and 'ev_ebitda' keys
    """
    return [
        {"Industry": industry, "EV/EBITDA": multiple if multiple else "N/A"}
        for industry, multiple in sorted(DAMODARAN_EV_EBITDA_MULTIPLES.items())
    ]


# Damodaran source URL for reference
DAMODARAN_SOURCE_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/vebitda.html"
DAMODARAN_DATA_DATE = "January 2026"
