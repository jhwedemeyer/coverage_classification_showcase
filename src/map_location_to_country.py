"""
Load all unique locations from all_locations.pkl, perform nominatim geocoding,
and store country names, location types, and all other nominatim data.

Only locations that are not already present in all_locations_geocoded.pkl are
geocoded; new results are appended to the existing output files.
"""

import os
import pandas as pd
import pickle
import json
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

geolocator = Nominatim(user_agent="pik_international_events_research")

INPUT_PKL = "data/localisation/all_locations.pkl"
OUTPUT_PKL = "data/localisation/all_locations_geocoded.pkl"
OUTPUT_CSV = "data/localisation/all_locations_geocoded.csv"

# Write intermediate results to disk every N newly geocoded locations
CHECKPOINT_EVERY = 250

# Backoff (seconds) between attempts when the service is unreachable
RETRY_BACKOFF = [5, 15, 45, 120, 300, 600]


class GeocoderUnreachable(Exception):
    """Raised when Nominatim stays unreachable after all retries."""


def geocode_location(location_name):
    """
    Geocode a location using nominatim.

    A `None` return means Nominatim answered but found no match - that is a
    real result and gets recorded. Connection problems are retried with
    backoff and, if they persist, raise `GeocoderUnreachable` so the caller
    can stop the run instead of recording a bogus "no match".

    Parameters:
    -----------
    location_name : str
        The location name to geocode

    Returns:
    --------
    geopy.Location or None
        The nominatim result, or None if the location is unknown to Nominatim

    Raises:
    -------
    GeocoderUnreachable
        If the service could not be reached after all retries
    """
    for attempt, wait in enumerate(RETRY_BACKOFF, 1):
        try:
            return geolocator.geocode(location_name, timeout=10, addressdetails=True)
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            logger.warning(
                f"Service unreachable for '{location_name}' ({e}), "
                f"attempt {attempt}/{len(RETRY_BACKOFF)}, waiting {wait}s"
            )
            time.sleep(wait)
        except GeocoderServiceError as e:
            # Query-level rejection (malformed query, quota, auth): the location
            # itself is the problem, so record it as unresolvable and move on.
            logger.error(f"Service error for '{location_name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error geocoding '{location_name}': {e}")
            return None

    raise GeocoderUnreachable(
        f"Nominatim unreachable after {len(RETRY_BACKOFF)} attempts "
        f"(last location: '{location_name}')"
    )


def extract_nominatim_info(location_object):
    """
    Extract relevant information from nominatim location object.
    
    Parameters:
    -----------
    location_object : geopy.Location
        The nominatim location object
        
    Returns:
    --------
    dict
        Dictionary containing extracted information
    """
    if location_object is None:
        return {
            'country': None,
            'country_code': None,
            'location_type': None,
            'latitude': None,
            'longitude': None,
            'address': None,
            'raw': None
        }
    
    address = location_object.raw.get('address', {})
    
    location_type = location_object.raw.get('type', None)
    
    # Extract country and country_code
    country = address.get('country', None)
    country_code = address.get('country_code', None)
    
    return {
        'country': country,
        'country_code': country_code,
        'location_type': location_type,
        'latitude': location_object.latitude,
        'longitude': location_object.longitude,
        'address': location_object.address,
        'raw': location_object.raw
    }


def load_existing_results():
    """
    Load already geocoded locations, if any.

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by 'location_normalized' (empty if no output exists yet)
    """
    if not os.path.exists(OUTPUT_PKL):
        logger.info(f"No existing output at {OUTPUT_PKL}, starting from scratch")
        return pd.DataFrame()

    existing = pd.read_pickle(OUTPUT_PKL)
    if existing.index.name != 'location_normalized':
        existing = existing.set_index('location_normalized')
    logger.info(f"Loaded {len(existing)} already geocoded locations from {OUTPUT_PKL}")
    return existing


def append_results(existing, new_records):
    """
    Combine previously geocoded locations with newly geocoded records.

    Parameters:
    -----------
    existing : pd.DataFrame
        Previously geocoded locations, indexed by 'location_normalized'
    new_records : list of dict
        Newly geocoded records

    Returns:
    --------
    pd.DataFrame
        Combined results indexed by 'location_normalized'
    """
    if not new_records:
        return existing

    new = pd.DataFrame(new_records).set_index('location_normalized')
    if len(existing) == 0:
        return new
    # New records only cover locations missing from `existing`, so nothing is overwritten
    return pd.concat([existing, new])


def save_results(results):
    """
    Write the combined results to pickle and CSV, replacing the files atomically
    so an interrupted write cannot destroy previous results.
    """
    tmp_pkl = OUTPUT_PKL + ".tmp"
    results.to_pickle(tmp_pkl)
    os.replace(tmp_pkl, OUTPUT_PKL)

    tmp_csv = OUTPUT_CSV + ".tmp"
    results.drop(columns=['raw_nominatim_data']).to_csv(tmp_csv)
    os.replace(tmp_csv, OUTPUT_CSV)

    logger.info(f"✓ Saved {len(results)} records to {OUTPUT_PKL} and {OUTPUT_CSV}")


def main():
    """Main processing function."""

    logger.info("Loading all_locations.pkl...")
    try:
        all_locations = pd.read_pickle(INPUT_PKL)
        logger.info(f"Loaded {len(all_locations)} total location records")
    except FileNotFoundError:
        logger.error(f"File not found: {INPUT_PKL}")
        return
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        return

    # Get unique locations (case-insensitive)
    all_locations['location_lower'] = all_locations['location'].str.lower()
    unique_locations = all_locations['location_lower'].unique()
    logger.info(f"Found {len(unique_locations)} unique locations")

    # Skip locations that were geocoded in a previous run
    existing = load_existing_results()
    known = set(existing.index) if len(existing) else set()
    pending = [loc for loc in unique_locations if loc not in known]
    logger.info(
        f"{len(unique_locations) - len(pending)} locations already known, "
        f"{len(pending)} to geocode"
    )

    if not pending:
        logger.info("Nothing to do - all locations are already geocoded")
        results = existing
    else:
        # Map normalized -> original case once instead of scanning per location
        original_case = (
            all_locations.drop_duplicates('location_lower')
            .set_index('location_lower')['location']
        )

        results_list = []
        results = existing

        # Geocode each pending location
        logger.info("Starting geocoding process...")
        try:
            for i, loc in enumerate(pending, 1):
                if i % 50 == 0:
                    logger.info(f"Processing location {i}/{len(pending)}: {loc}")

                original_loc = original_case[loc]

                result = geocode_location(original_loc)

                info = extract_nominatim_info(result)

                record = {
                    'location_normalized': loc,
                    'location_original_case': original_loc,
                    'country': info['country'],
                    'country_code': info['country_code'],
                    'location_type': info['location_type'],
                    'latitude': info['latitude'],
                    'longitude': info['longitude'],
                    'address': info['address']
                }
                # Store raw data as JSON string to avoid indexing issues
                try:
                    record['raw_nominatim_data'] = json.dumps(info['raw']) if info['raw'] else None
                except (TypeError, ValueError):
                    record['raw_nominatim_data'] = None

                results_list.append(record)

                # Periodically persist progress so a crash does not lose work
                if i % CHECKPOINT_EVERY == 0:
                    results = append_results(existing, results_list)
                    save_results(results)

                # Rate limiting: Nominatim requests limit
                time.sleep(1)
        except (GeocoderUnreachable, KeyboardInterrupt) as e:
            # Persist what we have; the remaining locations stay pending and are
            # picked up by the next run instead of being stored as "no match".
            logger.warning(f"Stopping early: {e}")
            save_results(append_results(existing, results_list))
            logger.warning(
                f"{len(results_list)} locations saved, "
                f"{len(pending) - len(results_list)} still pending - rerun to resume"
            )
            return

        # Append the new records to the previously known ones and save
        results = append_results(existing, results_list)
        save_results(results)

    logger.info("\n" + "="*60)
    logger.info("GEOCODING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total unique locations in output: {len(results)}")
    logger.info(f"Newly geocoded in this run: {len(results) - len(existing)}")
    logger.info(f"Locations with country data: {results['country'].notna().sum()}")
    logger.info(f"Locations with country_code: {results['country_code'].notna().sum()}")
    logger.info(f"Locations with type info: {results['location_type'].notna().sum()}")
    logger.info(f"Locations with coordinates: {results['latitude'].notna().sum()}")
    logger.info("\nTop 20 countries by frequency:")
    country_counts = results['country'].value_counts().head(20)
    for country, count in country_counts.items():
        logger.info(f"  {country}: {count}")
    logger.info("\nLocation types found:")
    type_counts = results['location_type'].value_counts().head(20)
    for loc_type, count in type_counts.items():
        logger.info(f"  {loc_type}: {count}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
