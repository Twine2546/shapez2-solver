"""
Blueprint downloader for Shapez Vortex community blueprints.

Downloads blueprints from the community-vortex.shapez2.com API
for use in ML training datasets.
"""

import json
import time
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import sqlite3


@dataclass
class DownloadedBlueprint:
    """A downloaded blueprint from Shapez Vortex."""
    id: str
    title: str
    blueprint_code: str  # The SHAPEZ2-X-... code
    blueprint_type: str  # 'Island' or 'Building'
    building_count: int
    download_count: int
    view_count: int
    creator: str
    tags: List[str]
    created: str
    buildings: Dict[str, int]  # Building type counts
    cost: int
    raw_data: Dict[str, Any] = field(default_factory=dict)


class VortexAPI:
    """Client for the Shapez Vortex PocketBase API."""

    BASE_URL = "https://vortex.shapez2.io/api"
    BLUEPRINTS_COLLECTION = "f96llnpqjo2l87m"

    def __init__(self, delay_between_requests: float = 0.5):
        """
        Initialize the API client.

        Args:
            delay_between_requests: Seconds to wait between API calls (be nice to the server)
        """
        self.delay = delay_between_requests
        self._last_request_time = 0

    def _make_request(self, url: str) -> Dict[str, Any]:
        """Make a rate-limited request to the API."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        self._last_request_time = time.time()

        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Shapez2-Solver-ML/1.0 (research)',
                'Accept': 'application/json',
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.reason}")
            raise
        except urllib.error.URLError as e:
            print(f"URL Error: {e.reason}")
            raise

    def fetch_blueprints(
        self,
        page: int = 1,
        per_page: int = 50,
        sort_by: str = "-downloadCount",
        filter_tag: Optional[str] = None,
        blueprint_type: Optional[str] = None,  # 'Island' or 'Building'
    ) -> List[DownloadedBlueprint]:
        """
        Fetch a page of blueprints from the API.

        Args:
            page: Page number (1-indexed)
            per_page: Results per page (max 200)
            sort_by: Sort field (prefix with - for descending)
                     Options: created, downloadCount, viewCount, buildingCount
            filter_tag: Optional tag to filter by
            blueprint_type: Optional type filter ('Island' or 'Building')

        Returns:
            List of DownloadedBlueprint objects
        """
        url = f"{self.BASE_URL}/collections/{self.BLUEPRINTS_COLLECTION}/records"
        url += f"?page={page}&perPage={per_page}&sort={sort_by}"

        # Add filters
        filters = []
        if filter_tag:
            filters.append(f'tags~"{filter_tag}"')
        if blueprint_type:
            filters.append(f'type="{blueprint_type}"')
        if filters:
            url += f"&filter={urllib.parse.quote(' && '.join(filters))}"

        data = self._make_request(url)

        blueprints = []
        for item in data.get('items', []):
            try:
                bp = self._parse_blueprint_item(item)
                if bp and bp.blueprint_code:  # Only include if has code
                    blueprints.append(bp)
            except Exception as e:
                print(f"Warning: Failed to parse blueprint {item.get('id', 'unknown')}: {e}")

        return blueprints

    def _parse_blueprint_item(self, item: Dict[str, Any]) -> Optional[DownloadedBlueprint]:
        """Parse a raw API item into a DownloadedBlueprint."""
        blueprint_code = item.get('data', '')
        if not blueprint_code or not blueprint_code.startswith('SHAPEZ2'):
            return None

        # Extract creator name from expand if available
        creator = item.get('creator', '')
        expand = item.get('expand', {})
        if expand and 'creator' in expand:
            creator = expand['creator'].get('displayname', creator)

        # Extract tags
        tags = item.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]

        return DownloadedBlueprint(
            id=item.get('id', ''),
            title=item.get('title', 'Untitled'),
            blueprint_code=blueprint_code,
            blueprint_type=item.get('type', 'Unknown'),
            building_count=item.get('buildingCount', 0),
            download_count=item.get('downloadCount', 0),
            view_count=item.get('viewCount', 0),
            creator=creator,
            tags=tags,
            created=item.get('created', ''),
            buildings=item.get('buildings', {}),
            cost=item.get('cost', 0),
            raw_data=item,
        )

    def fetch_all_blueprints(
        self,
        max_blueprints: int = 1000,
        sort_by: str = "-downloadCount",
        filter_tag: Optional[str] = None,
        blueprint_type: Optional[str] = None,
        progress_callback=None,
    ) -> List[DownloadedBlueprint]:
        """
        Fetch multiple pages of blueprints.

        Args:
            max_blueprints: Maximum number of blueprints to fetch
            sort_by: Sort order
            filter_tag: Optional tag filter
            blueprint_type: Optional type filter
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of all fetched blueprints
        """
        all_blueprints = []
        page = 1
        per_page = 50

        while len(all_blueprints) < max_blueprints:
            try:
                blueprints = self.fetch_blueprints(
                    page=page,
                    per_page=per_page,
                    sort_by=sort_by,
                    filter_tag=filter_tag,
                    blueprint_type=blueprint_type,
                )

                if not blueprints:
                    break  # No more results

                all_blueprints.extend(blueprints)

                if progress_callback:
                    progress_callback(len(all_blueprints), max_blueprints)

                page += 1

            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break

        return all_blueprints[:max_blueprints]


class BlueprintStore:
    """SQLite storage for downloaded blueprints."""

    def __init__(self, db_path: str = "blueprints.db"):
        """Initialize the blueprint store."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blueprints (
                id TEXT PRIMARY KEY,
                title TEXT,
                blueprint_code TEXT,
                blueprint_type TEXT,
                building_count INTEGER,
                download_count INTEGER,
                view_count INTEGER,
                creator TEXT,
                tags TEXT,
                created TEXT,
                buildings TEXT,
                cost INTEGER,
                downloaded_at TEXT,
                raw_data TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS solver_results (
                blueprint_id TEXT,
                solver_mode TEXT,
                success INTEGER,
                solve_time REAL,
                num_belts INTEGER,
                throughput REAL,
                error_message TEXT,
                solved_at TEXT,
                features TEXT,
                FOREIGN KEY (blueprint_id) REFERENCES blueprints(id),
                PRIMARY KEY (blueprint_id, solver_mode)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_blueprints_type
            ON blueprints(blueprint_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_blueprints_count
            ON blueprints(building_count)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_solver_success
            ON solver_results(success)
        ''')

        conn.commit()
        conn.close()

    def save_blueprint(self, bp: DownloadedBlueprint):
        """Save a blueprint to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO blueprints
            (id, title, blueprint_code, blueprint_type, building_count,
             download_count, view_count, creator, tags, created,
             buildings, cost, downloaded_at, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bp.id,
            bp.title,
            bp.blueprint_code,
            bp.blueprint_type,
            bp.building_count,
            bp.download_count,
            bp.view_count,
            bp.creator,
            json.dumps(bp.tags),
            bp.created,
            json.dumps(bp.buildings),
            bp.cost,
            datetime.now().isoformat(),
            json.dumps(bp.raw_data),
        ))

        conn.commit()
        conn.close()

    def save_blueprints(self, blueprints: List[DownloadedBlueprint]):
        """Save multiple blueprints to the database."""
        for bp in blueprints:
            self.save_blueprint(bp)

    def save_solver_result(
        self,
        blueprint_id: str,
        solver_mode: str,
        success: bool,
        solve_time: float,
        num_belts: int = 0,
        throughput: float = 0.0,
        error_message: str = "",
        features: Optional[Dict] = None,
    ):
        """Save a solver result for a blueprint."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO solver_results
            (blueprint_id, solver_mode, success, solve_time, num_belts,
             throughput, error_message, solved_at, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            blueprint_id,
            solver_mode,
            1 if success else 0,
            solve_time,
            num_belts,
            throughput,
            error_message,
            datetime.now().isoformat(),
            json.dumps(features) if features else None,
        ))

        conn.commit()
        conn.close()

    def get_blueprints(
        self,
        blueprint_type: Optional[str] = None,
        min_buildings: int = 0,
        max_buildings: int = 10000,
        limit: int = 1000,
        exclude_solved: bool = False,
        solver_mode: Optional[str] = None,
    ) -> List[DownloadedBlueprint]:
        """
        Retrieve blueprints from the database.

        Args:
            blueprint_type: Filter by type
            min_buildings: Minimum building count
            max_buildings: Maximum building count
            limit: Max results
            exclude_solved: Exclude blueprints already solved
            solver_mode: If exclude_solved, which solver mode to check

        Returns:
            List of DownloadedBlueprint objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT id, title, blueprint_code, blueprint_type, building_count,
                   download_count, view_count, creator, tags, created,
                   buildings, cost, raw_data
            FROM blueprints
            WHERE building_count >= ? AND building_count <= ?
        """
        params = [min_buildings, max_buildings]

        if blueprint_type:
            query += " AND blueprint_type = ?"
            params.append(blueprint_type)

        if exclude_solved and solver_mode:
            query += """
                AND id NOT IN (
                    SELECT blueprint_id FROM solver_results
                    WHERE solver_mode = ?
                )
            """
            params.append(solver_mode)

        query += " ORDER BY download_count DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        blueprints = []
        for row in rows:
            bp = DownloadedBlueprint(
                id=row[0],
                title=row[1],
                blueprint_code=row[2],
                blueprint_type=row[3],
                building_count=row[4],
                download_count=row[5],
                view_count=row[6],
                creator=row[7],
                tags=json.loads(row[8]) if row[8] else [],
                created=row[9],
                buildings=json.loads(row[10]) if row[10] else {},
                cost=row[11],
                raw_data=json.loads(row[12]) if row[12] else {},
            )
            blueprints.append(bp)

        return blueprints

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total blueprints
        cursor.execute("SELECT COUNT(*) FROM blueprints")
        stats['total_blueprints'] = cursor.fetchone()[0]

        # By type
        cursor.execute("""
            SELECT blueprint_type, COUNT(*) FROM blueprints
            GROUP BY blueprint_type
        """)
        stats['by_type'] = {row[0]: row[1] for row in cursor.fetchall()}

        # Building count distribution
        cursor.execute("""
            SELECT
                CASE
                    WHEN building_count < 10 THEN '0-9'
                    WHEN building_count < 50 THEN '10-49'
                    WHEN building_count < 100 THEN '50-99'
                    WHEN building_count < 500 THEN '100-499'
                    ELSE '500+'
                END as range,
                COUNT(*)
            FROM blueprints
            GROUP BY range
        """)
        stats['building_count_ranges'] = {row[0]: row[1] for row in cursor.fetchall()}

        # Solver results
        cursor.execute("""
            SELECT solver_mode, COUNT(*), SUM(success), AVG(solve_time)
            FROM solver_results
            GROUP BY solver_mode
        """)
        stats['solver_results'] = {
            row[0]: {
                'total': row[1],
                'successes': row[2] or 0,
                'avg_time': row[3] or 0
            }
            for row in cursor.fetchall()
        }

        conn.close()
        return stats

    def get_training_data(
        self,
        positive_only: bool = False,
        negative_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get training data: blueprint + solver result pairs.

        Args:
            positive_only: Only return successful solves
            negative_only: Only return failed solves

        Returns:
            List of dicts with blueprint and solver info
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT b.*, s.solver_mode, s.success, s.solve_time, s.num_belts,
                   s.throughput, s.error_message, s.features
            FROM blueprints b
            JOIN solver_results s ON b.id = s.blueprint_id
        """

        if positive_only:
            query += " WHERE s.success = 1"
        elif negative_only:
            query += " WHERE s.success = 0"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                'blueprint': {
                    'id': row[0],
                    'title': row[1],
                    'blueprint_code': row[2],
                    'blueprint_type': row[3],
                    'building_count': row[4],
                    'download_count': row[5],
                    'view_count': row[6],
                    'creator': row[7],
                    'tags': json.loads(row[8]) if row[8] else [],
                    'created': row[9],
                    'buildings': json.loads(row[10]) if row[10] else {},
                    'cost': row[11],
                },
                'solver': {
                    'mode': row[14],
                    'success': bool(row[15]),
                    'solve_time': row[16],
                    'num_belts': row[17],
                    'throughput': row[18],
                    'error_message': row[19],
                    'features': json.loads(row[20]) if row[20] else None,
                },
                'label': 'positive' if bool(row[15]) else 'negative',
            })

        return results


def download_blueprints(
    max_count: int = 500,
    db_path: str = "blueprints.db",
    tags: Optional[List[str]] = None,
) -> int:
    """
    Download blueprints from Shapez Vortex.

    Args:
        max_count: Maximum blueprints to download
        db_path: Database path for storage
        tags: Optional list of tags to download (downloads each tag separately)

    Returns:
        Number of blueprints downloaded
    """
    api = VortexAPI(delay_between_requests=0.3)
    store = BlueprintStore(db_path)

    total_downloaded = 0

    def progress(current, total):
        print(f"  Downloaded {current}/{total} blueprints...")

    if tags:
        # Download each tag separately
        per_tag = max_count // len(tags)
        for tag in tags:
            print(f"\nDownloading blueprints with tag: {tag}")
            blueprints = api.fetch_all_blueprints(
                max_blueprints=per_tag,
                filter_tag=tag,
                progress_callback=progress,
            )
            store.save_blueprints(blueprints)
            total_downloaded += len(blueprints)
            print(f"  Saved {len(blueprints)} blueprints with tag '{tag}'")
    else:
        # Download most popular
        print("\nDownloading most popular blueprints...")
        blueprints = api.fetch_all_blueprints(
            max_blueprints=max_count,
            sort_by="-downloadCount",
            progress_callback=progress,
        )
        store.save_blueprints(blueprints)
        total_downloaded = len(blueprints)

    print(f"\nTotal downloaded: {total_downloaded}")
    print(f"Database stats: {store.get_stats()}")

    return total_downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Shapez 2 blueprints")
    parser.add_argument("--count", type=int, default=500,
                       help="Max blueprints to download")
    parser.add_argument("--db", type=str, default="blueprints.db",
                       help="Database path")
    parser.add_argument("--tags", type=str, nargs="*",
                       help="Tags to filter by")

    args = parser.parse_args()

    download_blueprints(
        max_count=args.count,
        db_path=args.db,
        tags=args.tags,
    )
