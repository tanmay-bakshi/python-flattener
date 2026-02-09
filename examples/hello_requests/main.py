import json

import requests

from localmod import make_message


def main() -> None:
    """Run the demo app.
    """

    msg: str = make_message()
    print(msg)

    # Use requests just to prove it's importable and functional without network.
    req: requests.Request = requests.Request("GET", "https://example.com/path", params={"a": "1"})
    prepared: requests.PreparedRequest = req.prepare()

    payload: dict[str, str] = {
        "requests_version": requests.__version__,
        "prepared_url": prepared.url,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
