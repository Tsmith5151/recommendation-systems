import json
import requests
import flask
import argparse


def similarity(user_id: str, host: str = "0.0.0.0", port: int = 5000):

    """API call to flask app running on localhost
    and fetch top similar customers to the input customer(s)

    Parameters
    ----------
    user_id: str
        unique user id to fetch similar user profiles
    host: str (default = localhost)
        hostname for serving the Flask sever
    port: int
        port for Flask sever to listen on
    """
    url = f"http://{host}:{port}/api/similarity/"
    to_json = json.dumps({"user_handle": user_id})
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}
    response = requests.post(url, data=to_json, headers=headers)
    return response.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Recommendation Application")
    parser.add_argument(
        "--hostname",
        default="0.0.0.0",
        type=str,
        help="hostname for serving Flask application",
    )
    parser.add_argument(
        "--port",
        default=5000,
        type=int,
        help="port for serving Flask application",
    )
    args = parser.parse_args()
    similarity(args.user_id, args.hostname, args.port)
