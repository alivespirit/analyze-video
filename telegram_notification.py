import os
import asyncio
import logging
import time
import shutil
import json

from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InputMediaPhoto
import telegram.error
from telegram.helpers import escape_markdown

logger = logging.getLogger()

VIDEO_FOLDER = os.getenv("VIDEO_FOLDER")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- State for Grouping "No Motion" Messages ---
# These are safe to use as module-level globals because the executor has max_workers=1,
# ensuring sequential processing and preventing race conditions.
no_motion_group_message_id = None
no_motion_grouped_videos = []

# --- NEW: Add a lock for Telegram message grouping ---
telegram_lock = asyncio.Lock()

# Mapping of sent full video messages to their source file paths
video_message_map = {}
message_map_lock = asyncio.Lock()

# Persistent mapping storage
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMP_DIR = os.path.join(_SCRIPT_DIR, "temp")
_MAP_FILE_PATH = os.path.join(_TEMP_DIR, "message_map.json")
_GROUP_FILE_PATH = os.path.join(_TEMP_DIR, "no_motion_group.json")

def load_message_map_from_disk():
    try:
        if os.path.exists(_MAP_FILE_PATH):
            with open(_MAP_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        logger.warning(f"Failed to load message map: {e}")
    return {}

def save_message_map_to_disk():
    try:
        os.makedirs(_TEMP_DIR, exist_ok=True)
        # Prune to keep only the latest 400 entries
        try:
            if len(video_message_map) > 400:
                # preserve insertion order; pop oldest
                keys = list(video_message_map.keys())
                for k in keys[: len(video_message_map) - 200]:
                    video_message_map.pop(k, None)
        except Exception:
            pass
        with open(_MAP_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(video_message_map, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save message map: {e}")

# Initialize mapping from disk on import
video_message_map.update(load_message_map_from_disk())

# --- Persistent state for 'no motion' group ---
def load_group_state_from_disk():
    try:
        if os.path.exists(_GROUP_FILE_PATH):
            with open(_GROUP_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    mid = data.get("message_id")
                    vids = data.get("videos")
                    if isinstance(vids, list):
                        return {"message_id": mid, "videos": vids}
    except Exception as e:
        logger.warning(f"Failed to load no-motion group state: {e}")
    return {"message_id": None, "videos": []}

def save_group_state_to_disk(message_id, videos):
    try:
        os.makedirs(_TEMP_DIR, exist_ok=True)
        payload = {"message_id": message_id, "videos": videos}
        with open(_GROUP_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save no-motion group state: {e}")

# Initialize no-motion group state from disk on import
try:
    _grp = load_group_state_from_disk()
    no_motion_group_message_id = _grp.get("message_id")
    no_motion_grouped_videos = _grp.get("videos") or []
    if no_motion_group_message_id and not isinstance(no_motion_group_message_id, int):
        # Guard against unexpected types
        try:
            no_motion_group_message_id = int(no_motion_group_message_id)
        except Exception:
            no_motion_group_message_id = None
except Exception as e:
    logger.warning(f"Failed to initialize no-motion group state: {e}")

async def reaction_callback(update, context):
    """
    Handles message reaction updates. If a user reacts with 🤔 to a full video
    message sent via the "Глянути" button, copy that video to temp/training for
    further analysis.
    """
    mr = getattr(update, "message_reaction", None)
    if not mr:
        return

    try:
        new_reactions = getattr(mr, "new_reaction", None) or []
    except Exception:
        new_reactions = []
    # Also capture removed reactions (old_reaction) to handle suffix append on removal
    try:
        old_reactions = getattr(mr, "old_reaction", None) or []
    except Exception:
        old_reactions = []

    # Determine reaction types present
    has_thinking = False
    has_up = False
    has_down = False
    for r in new_reactions:
        emoji = getattr(r, "emoji", None)
        if emoji == "🤔":
            has_thinking = True
        elif emoji == "👍":
            has_up = True
        elif emoji == "👎":
            has_down = True
    # Detect removal of thumbs up/down
    removed_up = any(getattr(r, "emoji", None) == "👍" for r in old_reactions)
    removed_down = any(getattr(r, "emoji", None) == "👎" for r in old_reactions)

    message_id = getattr(mr, "message_id", None)
    chat = getattr(mr, "chat", None)
    chat_id = getattr(chat, "id", None)

    # Find mapped video path
    key = f"{chat_id}:{message_id}"
    entry = video_message_map.get(key)
    if isinstance(entry, str):
        # Backward compatibility: only path stored
        video_path = entry
        # Reconstruct base caption
        if video_path.startswith(os.path.join(VIDEO_FOLDER, '')):
            rel = video_path[len(os.path.join(VIDEO_FOLDER, '')):].replace(os.path.sep, '/')
        else:
            rel = os.path.basename(video_path)
        caption = f"Осьо відео `{escape_markdown(rel, version=2)}`"
        entry = {"path": video_path, "caption": caption}
        video_message_map[key] = entry
    else:
        video_path = entry.get("path") if entry else None
        caption = entry.get("caption") if entry else None
    if not video_path:
        try:
            all_rs = []
            all_rs.extend([r for r in new_reactions if getattr(r, "emoji", None)])
            all_rs.extend([r for r in old_reactions if getattr(r, "emoji", None)])
            emojis_str = ",".join([getattr(r, "emoji", "?") for r in all_rs]) or "unknown"
        except Exception:
            emojis_str = "unknown"
        logger.warning(f"[unknown] Reaction(s) {emojis_str} received for message {message_id}, but no video mapping found.")
        return

    file_basename = os.path.basename(video_path)

    # Determine parse mode for caption edits
    parse_mode = None
    if isinstance(entry, dict):
        parse_mode = entry.get("mode")
    if not parse_mode:
        parse_mode = 'MarkdownV2' if (caption and caption.startswith("Осьо відео ")) else 'Markdown'

    # Apply actions based on reactions
    new_caption = caption or ""
    if has_up:
        logger.info(f"[{file_basename}] Reaction detected: object went away.")
        if parse_mode == 'MarkdownV2':
            new_caption = (new_caption or "") + " - *Юху\\!*"
        else:
            new_caption = (new_caption or "") + " - *Юху!*"
    if has_down:
        logger.info(f"[{file_basename}] Reaction detected: object came back.")
        if parse_mode == 'MarkdownV2':
            new_caption = (new_caption or "") + " - *Ex\\.\\.\\.*"
        else:
            new_caption = (new_caption or "") + " - *Ex...*"
    if has_thinking:
        logger.info(f"[{file_basename}] Reaction detected: video marked for further analysis (via 🤔 reaction).")
        # Copy to TEMP_DIR/training
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            temp_dir = os.path.join(script_dir, "temp")
            training_dir = os.path.join(temp_dir, "training")
            os.makedirs(training_dir, exist_ok=True)
            dest_path = os.path.join(training_dir, file_basename)
            base, ext = os.path.splitext(file_basename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(training_dir, f"{base}_{counter}{ext}")
                counter += 1
            shutil.copy2(video_path, dest_path)
            logger.info(f"[{file_basename}] Copied to training: {dest_path}")
        except Exception as e:
            logger.error(f"[{file_basename}] Failed to copy video to training: {e}")
        # Append note
        if parse_mode == 'MarkdownV2':
            new_caption = (new_caption or "") + "\n_Глянемо\\.\\.\\._"
        else:
            new_caption = (new_caption or "") + "\n_Глянемо..._"

    # If a thumbs up/down was removed, append the 'Нєа...' suffix and log
    if removed_up or removed_down:
        logger.info(f"[{file_basename}] Reaction removed.")
        if parse_mode == 'MarkdownV2':
            new_caption = (new_caption or "") + " - _Нєа\\.\\.\\._"
        else:
            new_caption = (new_caption or "") + " - _Нєа..._"

    # Edit message caption if changed
    try:
        if new_caption and new_caption != caption:
            await context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=new_caption, parse_mode=parse_mode)
            # Persist updated caption
            async with message_map_lock:
                video_message_map[key] = {"path": video_path, "caption": new_caption}
                save_message_map_to_disk()
    except telegram.error.BadRequest as e:
        # Retry with escaped base portion; preserve suffix formatting
        try:
            base_only = caption or ""
            suffix_only = new_caption[len(base_only):] if new_caption and base_only and new_caption.startswith(base_only) else (new_caption or "")
            escaped_base = escape_markdown(base_only, version=(2 if parse_mode == 'MarkdownV2' else 1))
            updated_text2 = escaped_base + suffix_only
            await context.bot.edit_message_caption(chat_id=chat_id, message_id=message_id, caption=updated_text2, parse_mode=parse_mode)
            async with message_map_lock:
                video_message_map[key] = {"path": video_path, "caption": updated_text2, "mode": parse_mode}
                save_message_map_to_disk()
        except Exception as e2:
            logger.warning(f"[{file_basename}] Failed to edit caption after escape: {e2}", exc_info=True)
    except Exception as e:
        logger.warning(f"[{file_basename}] Unexpected error editing caption: {e}", exc_info=True)


async def wait_for_unlock(media_path: str, max_wait: int = 120, interval: int = 10, logger: logging.Logger = None, file_basename: str = "") -> bool:
    """Waits for a "*.lock" file to disappear up to max_wait seconds.

    Returns True if a lock was observed during waiting, False otherwise.
    """
    start = time.monotonic()
    lock_path = media_path + ".lock"
    saw_lock = False
    while os.path.exists(lock_path) and (time.monotonic() - start) < max_wait:
        if logger:
            logger.info(f"[{file_basename}] Waiting for lock file on {media_path} to be released...")
        saw_lock = True
        await asyncio.sleep(interval)

    if os.path.exists(lock_path) and logger:
        logger.warning(f"[{file_basename}] Lock file still exists after {max_wait} seconds. Proceeding anyway: {lock_path}")
    return saw_lock


async def delete_with_retries(media_path: str, retries: int = 5, delay: int = 10, logger: logging.Logger = None, file_basename: str = "") -> None:
    """Deletes a file with simple retry/backoff."""
    for attempt in range(1, retries + 1):
        try:
            os.remove(media_path)
            if logger:
                logger.info(f"[{file_basename}] Temporary media file deleted: {media_path}")
            return
        except FileNotFoundError:
            # Already gone, treat as success
            if logger:
                logger.info(f"[{file_basename}] Temporary media file already deleted: {media_path}")
            return
        except Exception as e:
            if logger:
                logger.warning(f"[{file_basename}] Failed to delete temporary media file {media_path} (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                await asyncio.sleep(delay)


async def cleanup_temp_media(media_path: str, file_path: str, logger: logging.Logger, file_basename: str) -> None:
    """Waits for optional lock then deletes temporary media file if different from original."""
    if media_path == file_path or not os.path.exists(media_path):
        return
    await asyncio.sleep(10)  # Initial delay before checking for lock
    saw_lock = await wait_for_unlock(media_path, max_wait=120, interval=10, logger=logger, file_basename=file_basename)
    if saw_lock:
        # Grace period after lock release
        await asyncio.sleep(10)
    await delete_with_retries(media_path, retries=5, delay=10, logger=logger, file_basename=file_basename)


async def button_callback(update, context):
    """
    Handles button presses from inline keyboards in Telegram messages.
    When a user clicks a "Глянути" button, this function sends the corresponding full video.

    Args:
        update (telegram.Update): The update object from the Telegram API.
        context (telegram.ext.ContextTypes.DEFAULT_TYPE): The context object.
    """
    query = update.callback_query
    await query.answer() # Acknowledge callback quickly

    # Parse callback data (view only)
    raw = query.data
    callback_file_rel = raw.replace('/', os.path.sep)
    file_path = os.path.join(VIDEO_FOLDER, callback_file_rel)
    file_basename = os.path.basename(file_path)

    logger.info(f"[{file_basename}] Button callback received for: {callback_file_rel}")

    if not os.path.exists(file_path):
        logger.error(f"[{file_basename}] Video file not found for callback: {file_path}")
        try:
            await query.edit_message_text(text=f"{query.message.text}\n\n_{escape_markdown(query.data.replace('\\', '/'), version=2)}: Відео файл не знайдено._", parse_mode='MarkdownV2')
        except Exception as edit_e:
            logger.error(f"[{file_basename}] Error editing message for not found file: {edit_e}", exc_info=True)
        return
    # Only view action remains

    # "Глянути" view button
    logger.info(f"[{file_basename}] Sending video from callback...")
    try:
        with open(file_path, 'rb') as video_file:
            sent_video_msg = await context.bot.send_video(
                chat_id=query.message.chat_id, video=video_file, parse_mode='MarkdownV2',
                caption=f"Осьо відео `{escape_markdown(query.data.replace('\\', '/'), version=2)}`", reply_to_message_id=query.message.message_id
            )
        try:
            key = f"{query.message.chat_id}:{sent_video_msg.message_id}"
            base_caption = f"Осьо відео `{escape_markdown(query.data.replace('\\', '/'), version=2)}`"
            async with message_map_lock:
                video_message_map[key] = {"path": file_path, "caption": base_caption, "mode": "MarkdownV2"}
                save_message_map_to_disk()
            logger.info(f"[{file_basename}] Mapped message {sent_video_msg.message_id} to video path (persisted).")
        except Exception as map_e:
            logger.warning(f"[{file_basename}] Failed to persist sent video mapping: {map_e}")
        logger.info(f"[{file_basename}] Video sent successfully from callback.")
    except FileNotFoundError:
        logger.error(f"[{file_basename}] Video file disappeared before sending from callback: {file_path}")
        try:
            await query.edit_message_text(text=f"{query.message.text}\n\n_{escape_markdown(query.data.replace('\\', '/'), version=2)} Помилка: Відео файл зник._", parse_mode='MarkdownV2')
        except Exception as edit_e:
            logger.warning(f"[{file_basename}] Failed to edit message after video disappeared: {edit_e}", exc_info=True)
    except Exception as e:
        logger.error(f"[{file_basename}] Failed to send video from callback: {e}", exc_info=True)
        pass


async def send_notifications(app, video_response, insignificant_frames, clip_path, file_path, file_basename, timestamp_text):
    """
    Sends Telegram notifications based on analysis results, including:
    - Animation or message with button for significant motion.
    - Grouped messages for insignificant/no motion events.
    - Media group of insignificant motion frames.
    - Cleanup of temporary media files.

    Args:
        app (telegram.ext.Application): The configured Telegram application instance.
        video_response (str): The caption/message to send.
        insignificant_frames (list[str]): Paths to insignificant frames to send as photos.
        clip_path (str|None): Path to the generated highlight clip, if any.
        file_path (str): Original video file path.
        file_basename (str): Basename of the original video file.
        timestamp_text (str): Short timestamp text used for buttons/captions.
    """
    global no_motion_group_message_id, no_motion_grouped_videos

    sent_message = None

    async with telegram_lock:
        # --- REFINED DECISION LOGIC ---
        is_significant_motion = clip_path is not None

        safe_video_folder = os.path.join(VIDEO_FOLDER, '')
        if file_path.startswith(safe_video_folder):
            callback_file = file_path[len(safe_video_folder):].replace(os.path.sep, '/')
        else:
            callback_file = file_basename

        if is_significant_motion:
            media_path = clip_path
            if not os.path.exists(media_path):
                logger.warning(f"[{file_basename}] Highlight clip not found, using original video.")
                media_path = file_path
            try:
                # Single button row: only "Глянути" (use reactions for actions)
                keyboard = [[
                    InlineKeyboardButton("Глянути", callback_data=callback_file)
                ]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                with open(media_path, 'rb') as animation_file:
                    sent_message = await app.bot.send_animation(
                        chat_id=CHAT_ID, animation=animation_file, caption=video_response,
                        reply_markup=reply_markup, parse_mode='Markdown'
                    )
                    logger.info(f"[{file_basename}] Animation sent successfully.")
                    try:
                        async with message_map_lock:
                            video_message_map[f"{CHAT_ID}:{sent_message.message_id}"] = {"path": file_path, "caption": video_response, "mode": "Markdown"}
                            save_message_map_to_disk()
                        logger.info(f"[{file_basename}] Mapped animation message {sent_message.message_id} to video path (persisted).")
                    except Exception as map_e:
                        logger.warning(f"[{file_basename}] Failed to persist animation mapping: {map_e}")
            except telegram.error.BadRequest as bad_request_error:
                logger.warning(f"[{file_basename}] BadRequest error: {bad_request_error}. Retrying with escaped Markdown.")
                try:
                    with open(media_path, 'rb') as animation_file:
                        sent_message = await app.bot.send_animation(
                            chat_id=CHAT_ID,
                            animation=animation_file,
                            caption=escape_markdown(video_response, version=1),
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                    logger.info(f"[{file_basename}] Animation sent successfully after escaping Markdown.")
                    try:
                        escaped_caption = escape_markdown(video_response, version=1)
                        async with message_map_lock:
                            video_message_map[f"{CHAT_ID}:{sent_message.message_id}"] = {"path": file_path, "caption": escaped_caption, "mode": "Markdown"}
                            save_message_map_to_disk()
                        logger.info(f"[{file_basename}] Mapped escaped animation message {sent_message.message_id} to video path (persisted).")
                    except Exception as map_e:
                        logger.warning(f"[{file_basename}] Failed to persist escaped animation mapping: {map_e}")
                except Exception as retry_error:
                    logger.error(f"[{file_basename}] Failed to send animation after escaping Markdown: {retry_error}. Status: error", exc_info=True)
                    logger.info(f"[{file_basename}] Sending plain message with button to Telegram...")
                    try:
                        sent_message = await app.bot.send_message(
                            chat_id=CHAT_ID,
                            text=video_response,
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        logger.info(f"[{file_basename}] Plain message with button sent successfully.")
                        try:
                            async with message_map_lock:
                                video_message_map[f"{CHAT_ID}:{sent_message.message_id}"] = {"path": file_path, "caption": video_response, "mode": "Markdown"}
                                save_message_map_to_disk()
                            logger.info(f"[{file_basename}] Mapped plain message {sent_message.message_id} to video path (persisted).")
                        except Exception as map_e:
                            logger.warning(f"[{file_basename}] Failed to persist plain message mapping: {map_e}")
                    except telegram.error.BadRequest as bad_request_error_fallback:
                        logger.warning(f"[{file_basename}] BadRequest error on fallback: {bad_request_error_fallback}. Retrying with escaped Markdown.")
                        try:
                            sent_message = await app.bot.send_message(
                                chat_id=CHAT_ID,
                                text=escape_markdown(video_response, version=1),
                                reply_markup=reply_markup,
                                parse_mode='Markdown'
                            )
                            logger.info(f"[{file_basename}] Message sent successfully after escaping Markdown.")
                            try:
                                escaped_text = escape_markdown(video_response, version=1)
                                async with message_map_lock:
                                    video_message_map[f"{CHAT_ID}:{sent_message.message_id}"] = {"path": file_path, "caption": escaped_text, "mode": "Markdown"}
                                    save_message_map_to_disk()
                                logger.info(f"[{file_basename}] Mapped escaped plain message {sent_message.message_id} to video path (persisted).")
                            except Exception as map_e:
                                logger.warning(f"[{file_basename}] Failed to persist escaped plain message mapping: {map_e}")
                        except Exception as e_final_fallback:
                            logger.error(f"[{file_basename}] Failed to send message after escaping Markdown: {e_final_fallback}", exc_info=True)
            except Exception as e:
                logger.error(f"[{file_basename}] Error sending animation: {e}. Status: error", exc_info=True)
                logger.info(f"[{file_basename}] Sending plain message with button to Telegram...")
                try:
                    sent_message = await app.bot.send_message(
                        chat_id=CHAT_ID,
                        text=video_response,
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                    logger.info(f"[{file_basename}] Plain message with button sent successfully.")
                    try:
                        async with message_map_lock:
                            video_message_map[f"{CHAT_ID}:{sent_message.message_id}"] = {"path": file_path, "caption": video_response, "mode": "Markdown"}
                            save_message_map_to_disk()
                        logger.info(f"[{file_basename}] Mapped plain message {sent_message.message_id} to video path (persisted).")
                    except Exception as map_e:
                        logger.warning(f"[{file_basename}] Failed to persist plain message mapping: {map_e}")
                except telegram.error.BadRequest as bad_request_error:
                    logger.warning(f"[{file_basename}] BadRequest error: {bad_request_error}. Retrying with escaped Markdown.")
                    try:
                        sent_message = await app.bot.send_message(
                            chat_id=CHAT_ID,
                            text=escape_markdown(video_response, version=1),
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        logger.info(f"[{file_basename}] Message sent successfully after escaping Markdown.")
                        try:
                            escaped_text = escape_markdown(video_response, version=1)
                            async with message_map_lock:
                                video_message_map[f"{CHAT_ID}:{sent_message.message_id}"] = {"path": file_path, "caption": escaped_text, "mode": "Markdown"}
                                save_message_map_to_disk()
                            logger.info(f"[{file_basename}] Mapped escaped plain message {sent_message.message_id} to video path (persisted).")
                        except Exception as map_e:
                            logger.warning(f"[{file_basename}] Failed to persist escaped plain message mapping: {map_e}")
                    except Exception as retry_error:
                        logger.error(f"[{file_basename}] Failed to send message after escaping Markdown: {retry_error}. Status: error", exc_info=True)
                except Exception as e_send:
                    logger.error(f"[{file_basename}] Failed to send plain message: {e_send}. Status: error", exc_info=True)
            finally:
                await cleanup_temp_media(media_path, file_path, logger, file_basename)

        else: # --- This block now handles ALL non-significant videos ---
            video_info = {'text': video_response, 'callback': callback_file, 'timestamp': timestamp_text}

            if no_motion_group_message_id and len(no_motion_grouped_videos) < 4:
                logger.info(f"[{file_basename}] Adding to existing insignificant message group.")
                no_motion_grouped_videos.append(video_info)
                # Persist state immediately so we can recover after restart
                try:
                    save_group_state_to_disk(no_motion_group_message_id, no_motion_grouped_videos)
                except Exception as e:
                    logger.warning(f"[{file_basename}] Failed to persist group state after append: {e}")

                # Build the updated message
                full_text = "\n".join([v['text'] for v in no_motion_grouped_videos])

                # --- Create a single row of buttons ---
                button_row = [InlineKeyboardButton(v['timestamp'], callback_data=v['callback']) for v in no_motion_grouped_videos]
                reply_markup = InlineKeyboardMarkup([button_row]) # Note the double brackets [[...]]

                try:
                    await app.bot.edit_message_text(
                        chat_id=CHAT_ID,
                        message_id=no_motion_group_message_id,
                        text=full_text,
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                    logger.info(f"[{file_basename}] Successfully edited message to extend group.")
                    try:
                        save_group_state_to_disk(no_motion_group_message_id, no_motion_grouped_videos)
                    except Exception as e:
                        logger.warning(f"[{file_basename}] Failed to persist group state after edit: {e}")
                except telegram.error.BadRequest as e:
                    if "message is not modified" in str(e).lower():
                        logger.info(f"[{file_basename}] Message was not modified, skipping edit.")
                    else:
                        logger.warning(f"[{file_basename}] Could not edit message: {e}. Retrying with escaped Markdown.")
                        try:
                            await app.bot.edit_message_text(
                                chat_id=CHAT_ID,
                                message_id=no_motion_group_message_id,
                                text=escape_markdown(full_text, version=1),
                                reply_markup=reply_markup,
                                parse_mode='Markdown'
                            )
                            logger.info(f"[{file_basename}] Message edited successfully after escaping Markdown.")
                            try:
                                save_group_state_to_disk(no_motion_group_message_id, no_motion_grouped_videos)
                            except Exception as e2:
                                logger.warning(f"[{file_basename}] Failed to persist group state after escaped edit: {e2}")
                        except Exception as retry_error:
                            logger.error(f"[{file_basename}] Failed to edit message after escaping Markdown: {retry_error}. Status: error", exc_info=True)
                            no_motion_group_message_id = None
                            no_motion_grouped_videos.clear()
                            try:
                                save_group_state_to_disk(no_motion_group_message_id, no_motion_grouped_videos)
                            except Exception as e3:
                                logger.warning(f"[{file_basename}] Failed to persist group state after clearing: {e3}")

            elif no_motion_group_message_id is None or len(no_motion_grouped_videos) >= 4:
                logger.info(f"[{file_basename}] Starting a new insignificant message group.")
                no_motion_grouped_videos = [video_info]

                # Create the first button for the new message
                button_row = [InlineKeyboardButton("Глянути", callback_data=v['callback']) for v in no_motion_grouped_videos]
                reply_markup = InlineKeyboardMarkup([button_row])

                try:
                    sent_message = await app.bot.send_message(
                        chat_id=CHAT_ID,
                        text=video_info['text'],
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                    no_motion_group_message_id = sent_message.message_id
                    logger.info(f"[{file_basename}] New group message sent. Message ID: {no_motion_group_message_id}")
                    try:
                        save_group_state_to_disk(no_motion_group_message_id, no_motion_grouped_videos)
                    except Exception as e:
                        logger.warning(f"[{file_basename}] Failed to persist group state after new group send: {e}")
                except telegram.error.BadRequest as e:
                    logger.warning(f"[{file_basename}] Could not send new group message: {e}. Retrying with escaped Markdown.")
                    try:
                        sent_message = await app.bot.send_message(
                            chat_id=CHAT_ID,
                            text=escape_markdown(video_info['text'], version=1),
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        no_motion_group_message_id = sent_message.message_id
                        logger.info(f"[{file_basename}] New group message sent successfully after escaping Markdown. Message ID: {no_motion_group_message_id}")
                        try:
                            save_group_state_to_disk(no_motion_group_message_id, no_motion_grouped_videos)
                        except Exception as e2:
                            logger.warning(f"[{file_basename}] Failed to persist group state after escaped new group send: {e2}")
                    except Exception as retry_error:
                        logger.error(f"[{file_basename}] Failed to send new group message after escaping Markdown: {retry_error}. Status: error", exc_info=True)
                        no_motion_group_message_id = None
                        no_motion_grouped_videos.clear()
                        try:
                            save_group_state_to_disk(no_motion_group_message_id, no_motion_grouped_videos)
                        except Exception as e3:
                            logger.warning(f"[{file_basename}] Failed to persist group state after clearing: {e3}")
                except Exception as e_send:
                    logger.error(f"[{file_basename}] Failed to send new group message: {e_send}. Status: error", exc_info=True)
                    no_motion_group_message_id = None
                    no_motion_grouped_videos.clear()
                    try:
                        save_group_state_to_disk(no_motion_group_message_id, no_motion_grouped_videos)
                    except Exception as e4:
                        logger.warning(f"[{file_basename}] Failed to persist group state after clearing: {e4}")

        if insignificant_frames:
            logger.info(f"[{file_basename}] Found {len(insignificant_frames)} insignificant motion frames to send.")
            media_group = []
            frame_data = []
            for frame_path in insignificant_frames:
                try:
                    with open(frame_path, 'rb') as photo_file:
                        frame_data.append(photo_file.read())
                except Exception as e:
                    logger.error(f"[{file_basename}] Failed to read frame file {frame_path}: {e}")

            for data in frame_data:
                media_group.append(InputMediaPhoto(media=data))

            if media_group:
                try:
                    reply_to_id = None
                    if sent_message:
                        reply_to_id = sent_message.message_id
                    elif no_motion_group_message_id:
                        reply_to_id = no_motion_group_message_id

                    if reply_to_id:
                        await app.bot.send_media_group(
                            chat_id=CHAT_ID,
                            media=media_group,
                            reply_to_message_id=reply_to_id,
                            caption=f"_{timestamp_text}_ \U0001F4F8",
                            parse_mode='Markdown'
                        )
                        logger.info(f"[{file_basename}] Sent media group of {len(media_group)} insignificant frames as a reply.")
                    else:
                        logger.warning(f"[{file_basename}] No message ID to reply to. Sending media group without reply.")
                        await app.bot.send_media_group(chat_id=CHAT_ID, media=media_group, caption=f"_{timestamp_text}_ \U0001F4F8", parse_mode='Markdown')

                except Exception as e:
                    logger.error(f"[{file_basename}] Failed to send media group: {e}", exc_info=True)

            for frame_path in insignificant_frames:
                if os.path.exists(frame_path):
                    try:
                        os.remove(frame_path)
                        logger.info(f"[{file_basename}] Deleted temporary frame: {frame_path}")
                    except Exception as e:
                        logger.error(f"[{file_basename}] Failed to delete temporary frame {frame_path}: {e}")

        logger.info(f"[{file_basename}] Telegram interaction finished.")

