import asyncio, traceback
from .queue_manager import queue_manager
from ..apis.v1.trial import generate_plots

from ..core.database import  db_session_context
from ..apis.v1.schemas import TrialPlotRequest

async def batch_insert_messages():
    while True:
        try:
            await asyncio.sleep(1)

            await queue_manager.swap_queues()
            message_queue, secondary_queue = await queue_manager.get_queues()
            # print(secondary_queue)

            tasks = []

            # Create the DB session manually
            with db_session_context() as db:
                for trial_id, value_list in secondary_queue.items():
                    for item in value_list:
                        message = item["message"]
                        for date in message["weekly_cumulative_ranges"]:
                            trial_plot_data = {
                                    **message["trial_plot_request"],
                                    "cumulative_dates": date
                                }
                            trial_plot_request = TrialPlotRequest(**trial_plot_data)
                            tasks.append(generate_plots(trial_plot_request, db))

                if tasks:
                    await asyncio.gather(*tasks)

            await queue_manager.clear_secondary_queue()

        except Exception as e:
            traceback_data = traceback.format_exc()
            print(f"An error occurred in batch_insert_messages: {e} and the data is {secondary_queue} and the traceback data is {traceback_data}. Please check.")
            await queue_manager.clear_secondary_queue()