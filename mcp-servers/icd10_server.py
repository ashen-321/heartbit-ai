import asyncio
import logging
from fastmcp import FastMCP
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastMCP server
app = FastMCP("icd10")
started: bool = False


# HTTP request template used for all tools
async def _make_request(url: str):
    try:
        # Send request with query information
        async with httpx.AsyncClient(timeout=30.0) as client:
            logging.info(f"HTTP GET attempt to {url}")

            response = await client.get(url)
            response.raise_for_status()

            # Exit if there are no errors
            return response

    except httpx.HTTPStatusError:
        # Exit if sent a redirect error (standard behavior)
        return True

    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
        logging.warning(f"HTTP request attempt 1 failed: {type(e).__name__}: {e}")

        # Try again in [count] seconds
        attempt_count = 1
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Make 5 total request attempts
            while attempt_count <= 5:
                # Sleep for a bit
                sleep_time = attempt_count
                logging.info(f"Waiting {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

                try:
                    logging.debug(f"HTTP GET attempt to {url}")

                    response = await client.get(url)
                    response.raise_for_status()

                    # Exit if there are no errors
                    return response

                except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
                    # Try again if the request failed
                    logging.warning(f"HTTP request attempt {attempt_count} failed: {type(e).__name__}: {e}")
                    attempt_count += 1

                except Exception as e:
                    # Abort if an unusual error is caught
                    logging.error(f"Unexpected error in HTTP request: {e}")
                    return False

    except Exception as e:
        # Abort if an unusual error is caught
        logging.error(f"Unexpected error in HTTP request from NIH: {e}")
        return False


@app.tool(description="Search the National Institute of Health (NIH) database on International Classification of "
                      "Diseases (ICD) 10 codes for specificed conditions.The query input string will be used to "
                      "match to entries in NIH's database, so use the minimal amount of text while retaining the "
                      "intended meaning to avoid being overly specific and dropping relevant entries.")
async def get_icd10_code_basic(query: str) -> list:
    logging.info(f"Querying NIH ICD-10 for: {query}")

    # Make request
    nih_url = f'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code,name&terms={query}'
    response = await _make_request(nih_url)

    # Parse response
    response_body = response.json()[3]
    logging.info(f"ICD-10 query returned {response_body}")
    return response_body


@app.tool(description="Search the National Institute of Health (NIH) database on International Classification of "
                      "Diseases (ICD) 10 codes for specificed conditions. The query input string will be used to "
                      "match to entries in NIH's database, so use the minimal amount of text while retaining the "
                      "intended meaning to avoid being overly specific and dropping relevant entries. The max_list "
                      "input integer will be used to determine how many results are retrieved from the database, "
                      "with a default of 7.")
async def get_icd10_code_advanced(query: str, max_list: int = 7) -> list:
    logging.info(f"Querying NIH ICD-10 for {max_list} results related to: {query}")

    # Make request
    nih_url = f'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code,name&terms={query}&maxList={max_list}'
    response = await _make_request(nih_url)

    # Parse response
    response_body = response.json()[3]
    logging.info(f"ICD-10 query returned {response_body}")
    return response_body


def startup():
    global started

    if started:
        return

    started = True
    try:
        app.run(transport="streamable-http", host="0.0.0.0", port=8003)
    except KeyboardInterrupt:
        pass
    finally:
        logging.info('Server icd10 successfully shut down.')


if __name__ == '__main__':
    startup()
