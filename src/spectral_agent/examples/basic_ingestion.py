"""
Example: Basic Ingestion Workflow

This example demonstrates how to use the ingestion workflow to
search and download satellite imagery.

Prerequisites:
    - Configure USGS credentials for Landsat
    - Configure Copernicus credentials for Sentinel
    - See .env.example for required environment variables

Usage:
    python -m spectral_agent.examples.basic_ingestion
"""

import asyncio
import logging
from datetime import date

from rich.console import Console
from rich.table import Table

from spectral_agent.config import get_settings
from spectral_agent.graphs import run_ingestion_workflow
from spectral_agent.schemas.imagery import BoundingBox

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


async def main():
    """Run the ingestion workflow example."""

    console.print("\n[bold blue]🛰️ Spectral Agent - Ingestion Example[/bold blue]\n")

    # Check credentials
    settings = get_settings()

    available_satellites = []
    if settings.has_usgs_credentials():
        available_satellites.append("landsat")
        console.print("✅ USGS credentials configured (Landsat)")
    else:
        console.print("⚠️  USGS credentials not configured (Landsat unavailable)")

    if settings.has_copernicus_credentials():
        available_satellites.append("sentinel")
        console.print("✅ Copernicus credentials configured (Sentinel)")
    else:
        console.print("⚠️  Copernicus credentials not configured (Sentinel unavailable)")

    if not available_satellites:
        console.print(
            "\n[red]No satellite credentials configured. Please update .env file.[/red]"
        )
        return

    # Define area of interest (San Francisco Bay Area)
    bbox = BoundingBox(
        west=-122.5,
        south=37.5,
        east=-122.0,
        north=38.0,
    )

    console.print(f"\n[bold]Area of Interest:[/bold]")
    console.print(f"  West:  {bbox.west}°")
    console.print(f"  South: {bbox.south}°")
    console.print(f"  East:  {bbox.east}°")
    console.print(f"  North: {bbox.north}°")

    # Define date range
    start_date = date(2024, 1, 1)
    end_date = date(2024, 3, 31)

    console.print(f"\n[bold]Date Range:[/bold]")
    console.print(f"  Start: {start_date}")
    console.print(f"  End:   {end_date}")

    console.print(f"\n[bold]Satellites:[/bold] {', '.join(available_satellites)}")
    console.print(f"[bold]Max Cloud Cover:[/bold] 15%")
    console.print(f"[bold]Max Scenes:[/bold] 2 per satellite")

    # Run the workflow
    console.print("\n[yellow]Starting ingestion workflow...[/yellow]\n")

    try:
        result = await run_ingestion_workflow(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            satellites=available_satellites,
            max_cloud_cover=15.0,
            max_scenes=2,  # Limit for demo
        )

        # Display results
        if result.success:
            console.print(f"\n[green]✅ {result.message}[/green]\n")
        else:
            console.print(f"\n[red]❌ {result.message}[/red]\n")

        # Show Landsat results
        if result.landsat_results:
            table = Table(title="Landsat Downloads")
            table.add_column("Scene ID", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Size (MB)")
            table.add_column("Path")

            for dl in result.landsat_results:
                status = "✅" if dl.success else "❌"
                size = (
                    f"{dl.file_size_bytes / (1024*1024):.2f}"
                    if dl.file_size_bytes
                    else "-"
                )
                path = (
                    str(dl.file_path)[:50] + "..."
                    if dl.file_path and len(str(dl.file_path)) > 50
                    else str(dl.file_path or dl.error_message)
                )
                table.add_row(dl.scene_id[:30], status, size, path)

            console.print(table)

        # Show Sentinel results
        if result.sentinel_results:
            table = Table(title="Sentinel Downloads")
            table.add_column("Scene ID", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Size (MB)")
            table.add_column("Path")

            for dl in result.sentinel_results:
                status = "✅" if dl.success else "❌"
                size = (
                    f"{dl.file_size_bytes / (1024*1024):.2f}"
                    if dl.file_size_bytes
                    else "-"
                )
                path = (
                    str(dl.file_path)[:50] + "..."
                    if dl.file_path and len(str(dl.file_path)) > 50
                    else str(dl.file_path or dl.error_message)
                )
                table.add_row(dl.scene_id[:30], status, size, path)

            console.print(table)

        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total scenes found: {result.total_scenes_found}")
        console.print(f"  Downloaded: {result.total_scenes_downloaded}")
        console.print(f"  Duration: {result.total_download_time_seconds:.1f}s")

        if result.errors:
            console.print(f"\n[yellow]Warnings/Errors:[/yellow]")
            for err in result.errors:
                console.print(f"  - {err}")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Workflow failed")


if __name__ == "__main__":
    asyncio.run(main())
