import json
import requests
import logging
from parse import get_parser


def similarity(user_id: str, host: str = "0.0.0.0", port: int = 5000) -> json:

    """API call to flask app running on localhost
    and fetch top similar customers to the input customer(s)
    """
    url = f"http://{host}:{port}/api/similarity/"
    to_json = json.dumps({"user_handle": user_id})
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}
    response = requests.post(url, data=to_json, headers=headers)
    print(response.text)


if __name__ == "__main__":
    args = get_parser().parse_args()
    similarity(args.user_id, args.hostname, args.port)
