from sqladmin import ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request

from src.config import get_settings
from src.storage.models import Region


class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        settings = get_settings()
        if (
            settings.admin_password
            and form.get("username") == settings.admin_user
            and form.get("password") == settings.admin_password
        ):
            request.session["authenticated"] = True
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get("authenticated") is True


class RegionAdmin(ModelView, model=Region):
    column_list = [Region.id, Region.name, Region.active, Region.created_at]
    column_searchable_list = [Region.name]
    column_sortable_list = [Region.id, Region.name, Region.active, Region.created_at]
    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True
    name = "Region"
    name_plural = "Regions"
    icon = "fa-solid fa-map"
    form_include_pk = False
    column_labels = {
        Region.bbox: "Bounding Box (JSON: {min_lon, min_lat, max_lon, max_lat})",
    }
