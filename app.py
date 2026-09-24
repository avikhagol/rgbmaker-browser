import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="RGBMaker", css_file="custom.css")


@app.cell
def _():
    import sys

    import marimo as mo

    return mo, sys


@app.cell
def _(mo):
    mo.md(r"""
    # RGBMaker

    Make radio–optical composite images (ROR, IOU and optical RGB with TGSS/NVSS
    contours) for any object.

    > Note: Everything runs in your browser using
    [rgbmaker](https://pypi.org/project/rgbmaker/). The first run takes a little longer while Python loads.
    """)
    return


@app.cell
async def _(sys):
    # In the browser (Pyodide) install rgbmaker without its pinned dependencies;
    # the Pyodide builds of numpy/scipy/matplotlib/astropy are used instead.
    IN_BROWSER = "pyodide" in sys.modules
    if IN_BROWSER:
        import micropip

        await micropip.install(
            ["numpy", "scipy", "matplotlib", "astropy", "requests", "pyodide-http"]
        )
        await micropip.install(
            "rgbmaker", deps=False, index_urls="https://pypi.org/simple"
        )
        import pyodide_http

        pyodide_http.patch_all()
    return (IN_BROWSER,)


@app.cell
def _(mo):
    # Result images: show a JPEG preview and offer the full PNG as a lazy download.
    # In the browser build every image and eager download is inlined as base64, so
    # showing the PNG *and* offering it for download shipped it twice; noisy sky
    # images compress poorly as PNG, and two-panel figures went past marimo's
    # 8 MB output limit. The download callable only runs when the button is clicked.
    def image_with_download(png, filename, label, max_px=1600, quality=90):
        """Return [preview image, download button] for a PNG given as bytes."""
        import base64
        import io

        preview = png
        try:
            from PIL import Image

            with Image.open(io.BytesIO(png)) as source:
                rgb = source.convert("RGB")
            rgb.thumbnail((max_px, max_px), Image.LANCZOS)
            buf = io.BytesIO()
            # 4:4:4 chroma keeps thin coloured contour lines and labels sharp
            rgb.save(buf, format="JPEG", quality=quality, subsampling=0)
            preview = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass  # without Pillow, show the PNG itself
        return [
            mo.image(src=preview, width="100%"),
            mo.download(data=lambda: png, filename=filename,
                        mimetype="image/png", label=label),
        ]

    return (image_with_download,)


@app.cell(hide_code=True)
def _(IN_BROWSER, sys):
    import io
    import types

    # astroquery and regions aren't available in Pyodide. rgbmaker only uses
    # them for imagesopt=2 (VizieR catalogue overlays), so stub the imports.
    try:
        import astroquery  # noqa: F401
    except ImportError:
        class _Unavailable:
            URL = ""

        sys.modules["astroquery"] = types.ModuleType("astroquery")
        for _mod, _attr in [
            ("astroquery.skyview", "SkyView"),
            ("astroquery.nvas", "Nvas"),
            ("astroquery.vizier", "Vizier"),
        ]:
            _m = types.ModuleType(_mod)
            setattr(_m, _attr, _Unavailable())
            sys.modules[_mod] = _m
    try:
        import regions  # noqa: F401
    except ImportError:
        _m = types.ModuleType("regions")
        _m.PixCoord = _m.EllipsePixelRegion = None
        sys.modules["regions"] = _m

    import numpy as np
    import requests
    from astropy.io import fits
    from rgbmaker import RGBMaker

    import math
    import warnings

    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from scipy.ndimage import map_coordinates, uniform_filter

    # SkyView blocks cross-origin (browser) requests. Radio/IR/UV cutouts come
    # from CDS hips2fits; DSS2 comes from the STScI DSS server (the same
    # POSS-II/UKST plates SkyView uses), because the CDS DSS2 HiPS mosaics
    # contain horizontal streak artefacts.
    HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
    HIPS = {
        "TGSS ADR1": "astron.nl/P/tgssadr",
        "NVSS": "CDS/P/NVSS",
        "DSS2 Red": "CDS/P/DSS2/red",
        "DSS2 IR": "CDS/P/DSS2/NIR",
        "DSS2 Blue": "CDS/P/DSS2/blue",
        "WISE 22": "CDS/P/allWISE/W4",
        "WISE 12": "CDS/P/allWISE/W3",
        "WISE 4.6": "CDS/P/allWISE/W2",
        "WISE 3.4": "CDS/P/allWISE/W1",
        "GALEX Near UV": "CDS/P/GALEXGR6_7/NUV",
    }
    STSCI_DSS = "https://archive.stsci.edu/cgi-bin/dss_search"
    STSCI = {
        "DSS2 Red": "poss2ukstu_red",
        "DSS2 IR": "poss2ukstu_ir",
        "DSS2 Blue": "poss2ukstu_blue",
    }
    STSCI_MAX_ARCMIN = 40  # larger fields fall back to hips2fits

    def grid_header(ra, dec, fov, n):
        """Same TAN grid hips2fits produces, so every survey lines up."""
        cdelt = math.degrees(2 * math.tan(math.radians(fov) / 2) / n)
        h = fits.Header()
        h.update(NAXIS=2, NAXIS1=n, NAXIS2=n,
                 CTYPE1="RA---TAN", CTYPE2="DEC--TAN",
                 CRPIX1=n / 2, CRPIX2=n / 2, CRVAL1=ra, CRVAL2=dec,
                 CDELT1=-cdelt, CDELT2=cdelt, RADESYS="ICRS", LONPOLE=180.0)
        return h

    def fetch_stsci(survey, ra, dec, fov, n):
        size = fov * 60 * 1.2  # arcmin, margin for plate rotation
        resp = requests.get(STSCI_DSS, timeout=120, params=dict(
            v=survey, r=ra, d=dec, e="J2000", h=size, w=size, f="fits"))
        resp.raise_for_status()
        src = fits.open(io.BytesIO(resp.content))[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            src_wcs = WCS(src.header)
        header = grid_header(ra, dec, fov, n)
        tgt_wcs = WCS(header)
        # anti-alias when the output pixels are coarser than the plate pixels
        src_scale = float(np.mean(proj_plane_pixel_scales(src_wcs)))  # deg
        ratio = abs(header["CDELT2"]) / src_scale
        data = src.data.astype("float64")
        if ratio > 1.5:
            data = uniform_filter(data, size=int(round(ratio)))
        yy, xx = np.mgrid[0:n, 0:n]
        px, py = src_wcs.world_to_pixel(tgt_wcs.pixel_to_world(xx, yy))
        out = map_coordinates(data, [py, px], order=1, mode="nearest")
        return fits.HDUList([fits.PrimaryHDU(out, header=header)])

    def fetch_hips(hips, ra, dec, fov, n):
        resp = requests.get(HIPS2FITS, timeout=120, params=dict(
            hips=hips, ra=ra, dec=dec, fov=fov, width=n, height=n,
            projection="TAN", coordsys="icrs", format="fits"))
        resp.raise_for_status()
        hdul = fits.open(io.BytesIO(resp.content))
        hdul[0].data = np.nan_to_num(hdul[0].data.astype("float64"))
        return hdul

    def _get_imgl_hips(self, cals):
        c, svy, r, queue, ind, _sam = cals
        icrs = c.icrs
        ra, dec, n = icrs.ra.deg, icrs.dec.deg, int(self.px)
        fov = 2 * r.to("deg").value  # astroquery SkyView: size = 2 * radius
        try:
            if svy in STSCI and fov * 60 <= STSCI_MAX_ARCMIN:
                queue[ind] = [fetch_stsci(STSCI[svy], ra, dec, fov, n)]
            elif svy in HIPS:
                queue[ind] = [fetch_hips(HIPS[svy], ra, dec, fov, n)]
            else:
                queue[ind] = []
        except Exception as e:
            print(f"{svy} not found ({e})")
            queue[ind] = []
        return queue

    RGBMaker._get_imgl_pool = _get_imgl_hips

    from rgbmaker.fetch import query

    _ = IN_BROWSER
    return (
        HIPS,
        STSCI,
        STSCI_MAX_ARCMIN,
        fetch_hips,
        fetch_stsci,
        grid_header,
        np,
        query,
        requests,
    )


@app.cell(hide_code=True)
def _(HIPS, STSCI, STSCI_MAX_ARCMIN, fetch_hips, fetch_stsci, query, requests):
    # Downloads are cached for as long as the page is open, keyed on the inputs
    # that change the data (target, radius, image size, survey). Changing only
    # plot settings then redraws without fetching again. This cell never re-runs,
    # so the caches survive form submissions.
    from functools import lru_cache

    from astropy.coordinates import SkyCoord as _SkyCoord

    def _freeze(hdu):
        hdu.data.setflags(write=False)  # shared between runs: never modify in place
        return hdu

    @lru_cache(maxsize=32)
    def cached_resolve(position):
        c = _SkyCoord.from_name(position, frame="fk5").icrs
        return c.ra.deg, c.dec.deg, c.to_string("hmsdms")

    @lru_cache(maxsize=32)
    def cached_background(name, ra, dec, fov, n):
        if name in STSCI and fov * 60 <= STSCI_MAX_ARCMIN:
            return _freeze(fetch_stsci(STSCI[name], ra, dec, fov, n)[0])
        return _freeze(fetch_hips(HIPS[name], ra, dec, fov, n)[0])

    @lru_cache(maxsize=128)
    def cached_hips(hips, ra, dec, fov, n):
        return _freeze(fetch_hips(hips, ra, dec, fov, n)[0])

    @lru_cache(maxsize=128)
    def cached_tap(query_text):
        r = requests.get(
            "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
            params=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="json", QUERY=query_text),
            timeout=60,
        )
        r.raise_for_status()
        return tuple(tuple(row) for row in r.json()["data"])

    # RGB mode: the whole rgbmaker query, keyed on all of its arguments. rgbmaker
    # reports failures as a returned warning, so only successful results are kept.
    _query_cache = {}

    def cached_query(**kwargs):
        key = tuple(sorted(kwargs.items()))
        if key in _query_cache:
            return _query_cache[key]
        result = query(**kwargs)
        if result[0] == "success" and result[1]:
            if len(_query_cache) >= 16:
                _query_cache.pop(next(iter(_query_cache)))
            _query_cache[key] = result
        return result

    return (
        cached_background,
        cached_hips,
        cached_query,
        cached_resolve,
        cached_tap,
    )


@app.cell
def _(mo):
    MODES = ["RGB images (ROR · IOU · Optical)", "Composite contours", "Custom"]
    mode = mo.ui.radio(options=MODES, value=MODES[0], inline=True)
    mo.md(f"**What to make:** {mode}")
    return MODES, mode


@app.cell
def _(MODES, cform, form, mode):
    # the form for the selected mode, always directly under the mode selector
    # (Custom mode is laid out by its own section cells just below)
    {MODES[0]: form, MODES[1]: cform}.get(mode.value)
    return


@app.cell
def _(MODES, mo, mode, x_layout, x_single):
    # Custom · section 1: layout
    mo.stop(mode.value != MODES[2])
    mo.callout(mo.vstack([
        mo.md("#### 1 · Layout"),
        mo.hstack([x_layout, x_single if x_layout.value == "Single panel" else mo.md("")],
                  justify="start", gap=2),
    ]), kind="neutral")
    return


@app.cell
def _(FORM_BOX, MODES, mo, mode, x_target):
    # Custom · section 2: target
    mo.stop(mode.value != MODES[2])
    mo.md(f"""
    #### 2 · Target
    **Your name** (shown on the image)
    {x_target["name"]}

    **Target**: object name or FK5 J2000 coordinates
    {x_target["position"]}

    **Radius** (degrees, max 2) {x_target["radius"]} &nbsp; **Image size** {x_target["px"]}
    """).style(FORM_BOX)
    return


@app.cell
def _(FORM_BOX, MODES, mo, mode, show_rgb_panel, x_rgb):
    # Custom · section 3: RGB-C panel (colour composite + contours)
    mo.stop(mode.value != MODES[2] or not show_rgb_panel)
    mo.md(f"""
    #### 3 · RGB-C panel: colour composite with contours
    **Red** {x_rgb["r"]}
    {x_rgb["r_custom"]}

    **Green** {x_rgb["g"]}
    {x_rgb["g_custom"]}

    **Blue** {x_rgb["b"]}
    {x_rgb["b_custom"]}

    **Colour scaling** {x_rgb["scaling"]}

    **Brightness** {x_rgb["brightness"]} &nbsp; **Contrast** {x_rgb["contrast"]}

    **Contours**
    {x_rgb["contours"]}
    {x_rgb["contours_custom"]}
    """).style(FORM_BOX)
    return


@app.cell
def _(FORM_BOX, MODES, mo, mode, show_comp_panel, x_comp):
    # Custom · section 4: composite contour panel (background + contours)
    mo.stop(mode.value != MODES[2] or not show_comp_panel)
    mo.md(f"""
    #### 4 · Composite contour panel: background with contours
    **Background** {x_comp["bg"]}
    {x_comp["bg_custom"]}

    **Brightness** {x_comp["brightness"]} &nbsp; **Contrast** {x_comp["contrast"]}

    **Contours**
    {x_comp["contours"]}
    {x_comp["contours_custom"]}
    """).style(FORM_BOX)
    return


@app.cell
def _(FORM_BOX, MODES, mo, mode, x_cset):
    # Custom · section 5: contour levels and axes (shared by both panels)
    mo.stop(mode.value != MODES[2])
    mo.md(f"""
    #### 5 · Contour levels
    **Lowest contour** {x_cset["nsigma"]} × rms &nbsp; **Levels** {x_cset["nlev"]}

    {x_cset["floors"]}
    {x_cset["axes"]}

    Any HiPS ID from the [HiPS list](https://aladin.cds.unistra.fr/hips/list) that serves
    FITS tiles works in the custom boxes, e.g. `CDS/P/WENSS`.
    """).style(FORM_BOX)
    return


@app.cell
def _(FORM_BOX, MODES, mo, mode, x_run):
    # Custom · section 6: make the image
    mo.stop(mode.value != MODES[2])
    mo.vstack([mo.md("#### 6 · Make the image"), x_run]).style(FORM_BOX)
    return


@app.cell(hide_code=True)
def _(
    NO_CHANNEL,
    cached_resolve,
    draw_contour_set,
    grid_header,
    load_contours,
    load_image,
    mo,
    np,
    rgb_composite,
    set_custom,
    show_comp_panel,
    show_rgb_panel,
    stretch,
    survey_kind,
    x_comp,
    x_cset,
    x_rgb,
    x_run,
    x_target,
):
    # Custom mode: runs only when "Make custom image" is pressed; the result is
    # kept in state so it stays on screen while the settings are being changed.
    from io import BytesIO as _BytesIO

    import matplotlib.pyplot as _plt
    from astropy.wcs import WCS as _GridWCS
    from rgbmaker.imgplt import pl_RGB as _pl_RGB

    if x_run.value:
        _t, _r, _c, _s = x_target.value, x_rgb.value, x_comp.value, x_cset.value
        _panels = [p for p, on in (("rgb", show_rgb_panel), ("comp", show_comp_panel)) if on]
        _notes, _rows, _scaling = [], [], ""
        with mo.status.spinner(title="Resolving target and fetching cutouts…"):
            _ra, _dec, _target = cached_resolve(_t["position"].strip())
            _radius = min(float(_t["radius"]), 2.0)
            _fov, _n = 2 * _radius, int(_t["px"])
            _wcs = _GridWCS(grid_header(_ra, _dec, _fov, _n))  # every cutout is on this grid
            _cset = (_s["nsigma"], _s["nlev"], _s["floors"])

            def _finish(ax, title):
                ax.set_title(title, y=1, pad=-16, color="white")
                if _s["axes"]:
                    ax.axis("on")
                    ax.set_xlabel("RA (ICRS)")
                    ax.set_ylabel("Dec (ICRS)")
                ax.set_xlim(-0.5, _n - 0.5)
                ax.set_ylim(-0.5, _n - 0.5)
                ax.autoscale(False)

            _plt.ioff()
            _fig = _plt.figure(figsize=(10 * len(_panels), 10))
            for _i, _panel in enumerate(_panels, start=1):
                _ax = _fig.add_subplot(1, len(_panels), _i, projection=_wcs)
                if _panel == "rgb":
                    _chans, _kinds, _shorts = [], [], []
                    for _key, _name in (("r", "Red"), ("g", "Green"), ("b", "Blue")):
                        _choice = _r[_key]
                        _kinds.append(survey_kind(_choice))
                        if _choice == NO_CHANNEL:
                            _chans.append(None)
                            continue
                        try:
                            _data, _short = load_image(_choice, _r[_key + "_custom"], _ra, _dec, _fov, _n)
                            _chans.append(_data)
                            _shorts.append(_short)
                        except Exception as e:
                            _notes.append(f"RGB-C panel, {_name} channel ({_choice}): {e}")
                            _chans.append(None)
                    _img, _scaling = rgb_composite(_chans, _kinds, _r["scaling"],
                                                   _r["brightness"], _r["contrast"])
                    if _scaling == "none":
                        _img = np.zeros((_n, _n, 3))
                    _cont, _crows, _cnotes = load_contours(
                        _r["contours"], _r["contours_custom"], _ra, _dec, _fov, _n, _radius, *_cset)
                    _pl_RGB(_ax, _img, "", _t["name"], True)
                    _drawn = draw_contour_set(_ax, _cont)
                    _title = "RGB-C: " + "-".join(_shorts or ["none"])
                    _finish(_ax, _title + (" · contours: " + "-".join(_drawn) if _drawn else ""))
                    _label = "RGB-C"
                else:
                    try:
                        _bg, _bgshort = load_image(_c["bg"], _c["bg_custom"], _ra, _dec, _fov, _n)
                        _bg = stretch(_bg, 1.0, 100.0)
                    except Exception as e:
                        _notes.append(f"Composite panel background ({_c['bg']}): {e}")
                        _bg, _bgshort = np.zeros((_n, _n)), _c["bg"]
                    _bg = np.clip((_bg - 0.5) * _c["contrast"] + 0.5 + _c["brightness"], 0.0, 1.0)
                    _cont, _crows, _cnotes = load_contours(
                        _c["contours"], _c["contours_custom"], _ra, _dec, _fov, _n, _radius, *_cset)
                    _pl_RGB(_ax, _bg, "", _t["name"], True)
                    _drawn = draw_contour_set(_ax, _cont, extra_legend=_bgshort)
                    _finish(_ax, "-".join(_drawn + [_bgshort]))
                    _label = "Composite"
                _rows += [dict(panel=_label, **row) for row in _crows]
                _notes += [f"{_label} panel, {x}" for x in _cnotes]
            if len(_panels) == 2:
                _plt.subplots_adjust(wspace=0.01, hspace=0.01)
            _buf = _BytesIO()
            _fig.savefig(_buf, format="png", bbox_inches="tight",
                         pad_inches=0.05 if _s["axes"] else 0,
                         facecolor="white" if _s["axes"] else "black")
            _plt.close(_fig)
        set_custom(dict(png=_buf.getvalue(), notes=_notes, rows=_rows,
                        scaling=_scaling if "rgb" in _panels else "", target=_target))
    return


@app.cell
def _(MODES, get_custom, image_with_download, mo, mode):
    # Custom mode result (kept until the next "Make custom image")
    _res = get_custom()
    mo.stop(mode.value != MODES[2] or _res is None)
    _head = f"### {_res['target']}"
    if _res["scaling"]:
        _head += f"\n\nRGB-C colour scaling: **{_res['scaling']}**"
    _blocks = [
        mo.md(_head),
        *image_with_download(_res["png"], "custom_image.png",
                             "Download custom_image.png"),
    ]
    if _res["notes"]:
        _blocks.append(mo.callout(mo.md("\n".join(f"- {x}" for x in _res["notes"])), kind="warn"))
    if _res["rows"]:
        _blocks.append(mo.accordion({"Contour levels": mo.ui.table(_res["rows"], selection=None)}))
    mo.vstack(_blocks)
    return


@app.cell
def _(mo):
    form = (
        mo.md(r"""
        **Your name** (shown on the image)
        {name}

        **Target**: object name or FK5 J2000 coordinates, e.g. `M87`, `3C 33.1`, `14 09 48.86 -03 02 32.6`
        {position}

        **Radius** (degrees, max 2)
        {radius}

        **Image size** (pixels per survey cutout)
        {px}
        """)
        .batch(
            name=mo.ui.text(value="Avi"),
            position=mo.ui.text(value="speca", full_width=True),
            radius=mo.ui.number(start=0.01, stop=2.0, step=0.01, value=0.12),
            px=mo.ui.dropdown(options=["240", "480", "720"], value="480"),
        )
        .form(submit_button_label="Make RGB images", bordered=True)
    )
    return (form,)


@app.cell
def _(cached_query, form, mo):
    import base64
    import urllib.parse

    # Always define the outputs (older marimo versions run dependent cells
    # even when this one was stopped).
    images, info, otext, status = None, "", [], None
    if form.value is not None:
        _v = form.value
        with mo.status.spinner(
            title="Fetching TGSS, NVSS, DSS2, WISE and GALEX cutouts…",
            subtitle="This takes 15–30 seconds.",
        ):
            status, _uri, info, otext = cached_query(
                name=_v["name"],
                position=_v["position"],
                radius=_v["radius"],
                kind="base64",
                px=int(_v["px"]),
                imagesopt=1,
                annot=True,
            )

        images = []
        for _item in _uri:
            for _key, _src in _item.items():
                _b64 = urllib.parse.unquote(_src.split(",", 1)[1])
                images.append((_key, base64.b64decode(_b64)))
    return images, info, otext, status


@app.cell
def _(MODES, image_with_download, images, info, mo, mode, otext, status):
    _titles = {
        "img1": "ROR: TGSS · DSS2 Red · NVSS (NVSS and TGSS contours)",
        "img2": "IOU: WISE 22 · DSS2 Red · GALEX NUV, and optical DSS2 IR · Red · Blue (TGSS contours)",
    }
    if images is None:
        _out = None
    elif not images:
        _out = mo.callout(
            mo.md(f"**No images returned** ({status}): {info}"), kind="warn"
        )
    else:
        _details = {
            k: v
            for d in otext
            for k, v in d.items()
            if k not in ("survey_err", "exception")
        }
        _blocks = []
        for _i, (_key, _png) in enumerate(images, start=1):
            _blocks.append(mo.md(f"### {_titles.get(_key, _key)}"))
            _blocks.extend(
                image_with_download(
                    _png, f"output_{_i}.png", f"Download output_{_i}.png"
                )
            )
        _blocks.append(
            mo.accordion(
                {
                    "Details": mo.ui.table(
                        [{"item": k, "value": str(v)} for k, v in _details.items()],
                        selection=None,
                    )
                }
            )
        )
        _blocks.append(mo.md(f"<small>{info}</small>"))
        _out = mo.vstack(_blocks)
    _out if mode.value == MODES[0] else None
    return


@app.cell(hide_code=True)
def _(mo):
    # label: HiPS id, contour colour, short name for the title, rgbmaker's
    # fixed lowest contour (Jy/beam) where it has one, and whether to smooth
    # (rgbmaker smooths the high-resolution FIRST map the same way).
    CONTOUR_SURVEYS = {
        "TGSS ADR1 · 150 MHz (GMRT, 25″)": dict(hips="astron.nl/P/tgssadr", color="magenta", short="TGSS(GMRT)", floor=0.015),
        "NVSS · 1.4 GHz (VLA, 45″)": dict(hips="CDS/P/NVSS", color="cyan", short="NVSS(VLA)", floor=0.0015),
        "RACS-mid · 1.37 GHz (ASKAP, ~10″, Dec < +49°)": dict(hips="CSIRO/P/RACS/mid/I", color="yellow", short="RACS-mid(ASKAP)", smooth=True),
        "RACS-low · 888 MHz (ASKAP, ~15″, Dec < +41°)": dict(hips="CSIRO/P/RACS/low/I", color="orange", short="RACS-low(ASKAP)", smooth=True),
        "LoTSS DR3 · 144 MHz (LOFAR, 6″, north)": dict(hips="astron.nl/P/lotss_dr3_high", color="lime", short="LoTSS(LOFAR)", smooth=True),
        "LoTSS DR3 low-res · 144 MHz (LOFAR, 20″, north)": dict(hips="astron.nl/P/lotss_dr3_low", color="springgreen", short="LoTSS-low(LOFAR)"),
        "LoLSS DR1 · 54 MHz (LOFAR, 15″)": dict(hips="astron.nl/P/lolss_dr1", color="salmon", short="LoLSS(LOFAR)"),
        "VLSSr · 74 MHz (VLA, 75″)": dict(hips="CDS/P/VLSSr", color="red", short="VLSSr(VLA)"),
        "WENSS · 325 MHz (WSRT, 54″, north)": dict(hips="CDS/P/WENSS", color="deepskyblue", short="WENSS(WSRT)"),
        "SUMSS · 843 MHz (MOST, 45″, south)": dict(hips="CDS/P/SUMSS", color="gold", short="SUMSS(MOST)"),
    }
    BACKGROUNDS = ["DSS2 Red", "DSS2 IR", "DSS2 Blue", "WISE 3.4", "WISE 4.6", "WISE 12", "WISE 22", "GALEX Near UV"]
    _labels = list(CONTOUR_SURVEYS)
    cform = (
        mo.md(r"""
        **Your name** (shown on the image)
        {name}

        **Target**: object name or FK5 J2000 coordinates
        {position}

        **Radius** (degrees, max 2) {radius} &nbsp; **Image size** {px}

        **Layout** {layout}

        **Background image** {background}

        **Contour surveys** (choose any number)
        {surveys}

        **Extra contour survey** (optional): any HiPS ID from the
        [HiPS list](https://aladin.cds.unistra.fr/hips/list), e.g. `CDS/P/NVSS`
        {custom}

        **Lowest contour** {nsigma} × rms &nbsp; **Levels** {nlev}

        {floors}
        {catalog}
        {spidx_text}
        {axes}
        """)
        .batch(
            name=mo.ui.text(value="Avi"),
            position=mo.ui.text(value="speca", full_width=True),
            radius=mo.ui.number(start=0.01, stop=2.0, step=0.01, value=0.12),
            px=mo.ui.dropdown(options=["240", "480", "720"], value="480"),
            layout=mo.ui.dropdown(options=["Two panels", "Single panel"], value="Two panels"),
            background=mo.ui.dropdown(options=BACKGROUNDS, value="DSS2 Red"),
            surveys=mo.ui.multiselect(options=_labels, value=_labels[:3], full_width=True),
            custom=mo.ui.text(placeholder="e.g. CDS/P/WENSS", full_width=True),
            nsigma=mo.ui.number(start=1, stop=50, step=0.5, value=5),
            nlev=mo.ui.number(start=1, stop=12, step=1, value=4),
            floors=mo.ui.checkbox(value=True, label="TGSS and NVSS: use fixed lowest contour (15 and 1.5 mJy/beam) instead of N × rms"),
            catalog=mo.ui.checkbox(value=True, label="Mark TGSS and NVSS catalogue sources (VizieR)"),
            spidx_text=mo.ui.checkbox(value=False, label="Write spectral index values on the plot (otherwise sources are numbered; values are in the table below)"),
            axes=mo.ui.checkbox(value=False, label="Show RA/Dec axes"),
        )
        .form(submit_button_label="Make composite contour image", bordered=True)
    )
    return CONTOUR_SURVEYS, cform


@app.cell(hide_code=True)
def _(
    CONTOUR_SURVEYS,
    cached_background,
    cached_hips,
    cached_resolve,
    cached_tap,
    cform,
    mo,
    np,
):
    import io as _io

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS as _WCS
    from matplotlib.patches import Ellipse
    from rgbmaker.imgplt import pl_RGB, sqrt as rgb_sqrt
    from scipy.ndimage import gaussian_filter

    c_notes, c_rows, c_sources, c_spidx, c_target, contour_png = [], [], [], [], "", None

    def _levels(v, data, floor):
        good = data[np.isfinite(data) & (data != 0)]
        _, _, rms = sigma_clipped_stats(good, sigma=3, maxiters=5)
        low = v["nsigma"] * rms
        if floor is not None and v["floors"]:
            low = floor
        top = float(np.nanmax(data))
        if not np.isfinite(low) or low <= 0 or top <= low:
            return rms, None
        k = int(v["nlev"])
        lv = np.arange(low, top, (top - low) / k)[:k]  # linear, as rgbmaker
        return rms, lv

    def _inside(x, y, n):
        return -0.5 <= x <= n - 0.5 and -0.5 <= y <= n - 0.5

    if cform.value is not None:
        _v = cform.value
        two_panel = _v["layout"].startswith("Two")
        with mo.status.spinner(title="Resolving target and fetching cutouts…"):
            ra, dec, c_target = cached_resolve(_v["position"].strip())
            radius = min(float(_v["radius"]), 2.0)
            fov, n = 2 * radius, int(_v["px"])

            bg = cached_background(_v["background"], ra, dec, fov, n)
            wcs = _WCS(bg.header)
            pix_arcsec = abs(bg.header["CDELT2"]) * 3600
            bgd = np.nan_to_num(bg.data.astype(float))
            bgd = rgb_sqrt(bgd, scale_min=np.percentile(np.unique(bgd), 1.0),
                           scale_max=np.percentile(np.unique(bgd), 100.0))
            _bgshort = {"DSS2 Red": "DSS2R", "DSS2 IR": "DSS2IR", "DSS2 Blue": "DSS2B"}.get(
                _v["background"], _v["background"])
            _bgtitle = f"{_bgshort}(DSS)" if _v["background"].startswith("DSS") else _bgshort

            # ---- contour data -------------------------------------------------
            chosen = [(lab, CONTOUR_SURVEYS[lab]) for lab in _v["surveys"]]
            if _v["custom"].strip():
                _hid = _v["custom"].strip()
                chosen.append((_hid, dict(hips=_hid, color="white", short=_hid.split("/")[-1])))
            contours = []
            for lab, cfg in chosen:
                try:
                    hdu = cached_hips(cfg["hips"], ra, dec, fov, n)
                except Exception as e:
                    c_notes.append(f"{lab}: could not fetch ({e})")
                    continue
                data = hdu.data.astype(float)
                if np.count_nonzero(data) == 0:
                    c_notes.append(f"{lab}: target is outside the survey footprint")
                    continue
                if cfg.get("smooth"):
                    data = gaussian_filter(data, sigma=2 if radius < 0.12 else 1)
                rms, lv = _levels(_v, data, cfg.get("floor"))
                unit = hdu.header.get("BUNIT", "") or "Jy/beam?"
                c_rows.append(dict(survey=lab, rms=f"{rms:.3g} {unit}",
                                   levels=("none above threshold" if lv is None else
                                           ", ".join(f"{x:.3g}" for x in lv) + f" {unit}")))
                if lv is not None:
                    contours.append((cfg, data, lv))

            # ---- catalogues (VizieR) ------------------------------------------
            _circ = (f"1=CONTAINS(POINT('ICRS',RAJ2000,DEJ2000),"
                     f" CIRCLE('ICRS',{ra},{dec},{radius * 1.42}))")
            cats = []
            if _v["catalog"]:
                for cat, col, q in [
                    ("TGSS", "magenta", 'SELECT RAJ2000, DEJ2000, Maj, Min, PA, Stotal, e_Stotal FROM "J/A+A/598/A78/table3" WHERE ' + _circ),
                    ("NVSS", "cyan", 'SELECT RAJ2000, DEJ2000, MajAxis, MinAxis, PA, "S1.4", "e_S1.4" FROM "VIII/65/nvss" WHERE ' + _circ),
                ]:
                    try:
                        rows = cached_tap(q)
                    except Exception as e:
                        c_notes.append(f"{cat} catalogue: query failed ({e})")
                        continue
                    kept = []
                    for sra, sdec, maj, mnr, pa, s_, es in rows:
                        x, y = wcs.world_to_pixel(SkyCoord(sra * u.deg, sdec * u.deg))
                        if _inside(x, y, n):
                            kept.append((float(x), float(y), maj, mnr, pa))
                            c_sources.append(dict(catalogue=cat, id=len(kept), ra=round(sra, 5),
                                                  dec=round(sdec, 5), flux_mJy=s_, err_mJy=es,
                                                  maj_arcsec=maj, min_arcsec=mnr, pa_deg=pa))
                    cats.append((cat, col, kept))

            # de Gasperin, Intema & Frail 2018 (VizieR J/MNRAS/474/5008), the same
            # catalogue as rgbmaker's spidx file. S ∝ ν^α; Scode L = NVSS only, so α > x;
            # U = TGSS only, so α < x; I = island totals (table only).
            spx = []
            # spectral index is always shown
            try:
                rows = cached_tap('SELECT RAJ2000, DEJ2000, SpIndex, e_SpIndex, Scode, FtotNVSS, FtotTGSS '
                            'FROM "J/MNRAS/474/5008/spidxcat" WHERE ' + _circ)
            except Exception as e:
                rows = []
                c_notes.append(f"Spectral index catalogue: query failed ({e})")
            for sra, sdec, a, ea, code, snv, stg in rows:
                code = code or ""
                x, y = wcs.world_to_pixel(SkyCoord(sra * u.deg, sdec * u.deg))
                if not _inside(x, y, n):
                    continue
                pre = {"L": "> ", "U": "< "}.get(code, "")
                short = f"{pre}{a:.3f}"
                full = (f"α = {a:.2f} ± {ea:.2f}" if code in ("S", "M", "C") else
                        f"α {pre.strip()} {a:.2f}" if pre else f"α = {a:.2f} (island)")
                c_spidx.append(dict(label=f"α{len(c_spidx) + 1}", ra=round(sra, 5), dec=round(sdec, 5),
                                    alpha=full, Scode=code, S_NVSS_Jy=snv, S_TGSS_Jy=stg))
                if code != "I":
                    spx.append((float(x), float(y), short, full, len(c_spidx)))

            # ---- drawing helpers (rgbmaker styles) ----------------------------
            ell_scale = 1.0 / pix_arcsec  # catalogue FWHM axes at true size

            def draw_catalogues(ax):
                for cat, col, kept in cats:
                    for i, (x, y, maj, mnr, pa) in enumerate(kept, start=1):
                        if maj:
                            ax.add_patch(Ellipse((x, y), width=maj * ell_scale,
                                                 height=(mnr or maj) * ell_scale,
                                                 angle=90 + (pa or 0), facecolor="none",
                                                 edgecolor=col, lw=2))
                        if cat == "TGSS":
                            ax.annotate(i, xy=(x, y), xytext=(0, 0), textcoords="offset points", color=col)
                        else:
                            ax.annotate(i, xy=(x, y), xytext=(0, 0), textcoords="offset points",
                                        color=col, ha="right", va="top")

            def draw_spidx(ax, boxed_values_only):
                # value boxes if requested, otherwise the table id, same rgbmaker style
                kw = dict(arrowprops=dict(arrowstyle="->", ec=".5", relpos=(0.5, 0.5)),
                          bbox=dict(boxstyle="round", ec="none", fc="w"))
                for x, y, short, full, sid in spx:
                    if _v["spidx_text"]:
                        txt = short if boxed_values_only else full
                    else:
                        txt = f"α{sid}"
                    ax.annotate(txt, xy=(x, y), xytext=(1, -40), textcoords="offset points",
                                ha="right", va="top", color="black", **kw)

            def draw_contours(ax):
                handles, drawn = [], []
                for cfg, data, lv in contours:
                    ax.contour(data, lv, colors=cfg["color"])
                    handles.append(mpatches.Patch(color=cfg["color"], label=cfg["short"].split("(")[0]))
                    drawn.append(cfg["short"])
                handles.append(mpatches.Patch(color="white", label=_bgshort))
                ax.legend(handles=handles, labelcolor="linecolor", framealpha=0.0)
                return drawn

            def finish(ax, title):
                ax.set_title(title, y=1, pad=-16, color="white")
                if _v["axes"]:
                    ax.axis("on")
                    ax.set_xlabel("RA (ICRS)")
                    ax.set_ylabel("Dec (ICRS)")
                ax.set_xlim(-0.5, n - 0.5)
                ax.set_ylim(-0.5, n - 0.5)
                ax.autoscale(False)

            def credit(ax):
                if spx:
                    ax.text(0.5, 0.012, "α (147 MHz–1.4 GHz)",
                            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="white")

            plt.ioff()
            if two_panel:
                # rgbmaker imagesopt=2: left = catalogue sources + spectral index,
                # right = composite contours (background square-rooted twice, as rgbmaker does)
                fig = plt.figure(figsize=(20, 10))
                ax1 = fig.add_subplot(1, 2, 1, projection=wcs)
                pl_RGB(ax1, bgd, "", _v["name"], True)
                draw_catalogues(ax1)
                draw_spidx(ax1, boxed_values_only=True)
                credit(ax1)
                finish(ax1, f"TGSS(GMRT)-NVSS(VLA)-{_bgtitle}")

                bg2 = rgb_sqrt(bgd, scale_min=np.percentile(np.unique(bgd), 1.0),
                               scale_max=np.percentile(np.unique(bgd), 100.0))
                ax2 = fig.add_subplot(1, 2, 2, projection=wcs)
                pl_RGB(ax2, bg2, "", _v["name"], True)
                drawn = draw_contours(ax2)
                finish(ax2, "-".join(drawn + [_bgtitle]))
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
            else:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(1, 1, 1, projection=wcs)
                pl_RGB(ax, bgd, "", _v["name"], True)
                drawn = draw_contours(ax)
                draw_catalogues(ax)
                draw_spidx(ax, boxed_values_only=False)
                credit(ax)
                finish(ax, "-".join(drawn + [_bgtitle]))

            _buf = _io.BytesIO()
            fig.savefig(_buf, format="png", bbox_inches="tight",
                        pad_inches=0.05 if _v["axes"] else 0,
                        facecolor="white" if _v["axes"] else "black")
            plt.close(fig)
            contour_png = _buf.getvalue()
    return c_notes, c_rows, c_sources, c_spidx, c_target, contour_png


@app.cell(expand_output=True)
def _(
    MODES,
    c_notes,
    c_rows,
    c_sources,
    c_spidx,
    c_target,
    contour_png,
    image_with_download,
    mo,
    mode,
):
    mo.stop(contour_png is None or mode.value != MODES[1])
    _blocks = [
        mo.md(f"### Composite contours: {c_target}"),
        *image_with_download(contour_png, "composite_contours.png",
                             "Download composite_contours.png"),
    ]
    if c_notes:
        _blocks.append(mo.callout(mo.md("\n".join(f"- {x}" for x in c_notes)), kind="warn"))
    _acc = {"Contour levels": mo.ui.table(c_rows, selection=None)}
    if c_sources:
        _acc["Catalogue sources (TGSS ADR1, NVSS)"] = mo.ui.table(c_sources, selection=None)
    if c_spidx:
        _acc["Spectral index α, 147 MHz–1.4 GHz (de Gasperin+ 2018)"] = mo.vstack([
            mo.ui.table(c_spidx, selection=None),
            mo.md(
                "Source codes (de Gasperin, Intema & Frail 2018): **S** single, matched in both "
                "surveys · **M** several matched sources in one island (often double-lobed radio "
                "galaxies) · **C** matched, with unmatched detections in the island · **L** NVSS "
                "only (authors: “upper limit”; shown as α > x) · **U** TGSS only (“lower limit”; "
                "α < x) · **I** island totals. Fluxes are in Jy, rounded to 0.01 Jy as in the "
                "catalogue. Source: [VizieR J/MNRAS/474/5008](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/474/5008), "
                "[TGSS spectral index page](https://tgssadr.strw.leidenuniv.nl/doku.php?id=spidx)."
            ),
        ])
    _blocks.append(mo.accordion(_acc))
    mo.vstack(_blocks)
    return


@app.cell(hide_code=True)
def _(
    CONTOUR_SURVEYS,
    STSCI,
    STSCI_MAX_ARCMIN,
    cached_background,
    cached_hips,
    np,
):
    # Shared pieces for the Custom mode: survey list, image loading, stretch,
    # contour levels and drawing.
    import matplotlib.patches as _mpatches
    from astropy.stats import sigma_clipped_stats as _clipped_stats
    from scipy.ndimage import gaussian_filter as _gaussian

    CUSTOM_HIPS = "Custom HiPS ID…"
    NO_CHANNEL = "None"
    # label -> (short name for titles, HiPS ID). DSS2 comes from STScI (see setup).
    IMAGE_SURVEYS = {
        "DSS2 Red": ("DSS2R", "CDS/P/DSS2/red"),
        "DSS2 IR": ("DSS2IR", "CDS/P/DSS2/NIR"),
        "DSS2 Blue": ("DSS2B", "CDS/P/DSS2/blue"),
        "Pan-STARRS g": ("PS1-g", "CDS/P/PanSTARRS/DR1/g"),
        "Pan-STARRS r": ("PS1-r", "CDS/P/PanSTARRS/DR1/r"),
        "Pan-STARRS i": ("PS1-i", "CDS/P/PanSTARRS/DR1/i"),
        "Pan-STARRS z": ("PS1-z", "CDS/P/PanSTARRS/DR1/z"),
        "SDSS g": ("SDSS-g", "CDS/P/SDSS9/g"),
        "SDSS r": ("SDSS-r", "CDS/P/SDSS9/r"),
        "SDSS i": ("SDSS-i", "CDS/P/SDSS9/i"),
        "2MASS J": ("2MASS-J", "CDS/P/2MASS/J"),
        "2MASS H": ("2MASS-H", "CDS/P/2MASS/H"),
        "2MASS K": ("2MASS-K", "CDS/P/2MASS/K"),
        "WISE 3.4": ("WISE(3.4)", "CDS/P/allWISE/W1"),
        "WISE 4.6": ("WISE(4.6)", "CDS/P/allWISE/W2"),
        "WISE 12": ("WISE(12)", "CDS/P/allWISE/W3"),
        "WISE 22": ("WISE(22)", "CDS/P/allWISE/W4"),
        "GALEX Near UV": ("GALEX(NUV)", "CDS/P/GALEXGR6_7/NUV"),
        "GALEX Far UV": ("GALEX(FUV)", "CDS/P/GALEXGR6_7/FUV"),
    }
    for _lab, _cfg in CONTOUR_SURVEYS.items():
        IMAGE_SURVEYS[_lab] = (_cfg["short"], _cfg["hips"])

    def survey_kind(choice):
        """radio / uv / dss / other: used to pick rgbmaker's imagesopt=1 scaling."""
        if choice in CONTOUR_SURVEYS:
            return "radio"
        if choice.startswith("GALEX"):
            return "uv"
        if choice.startswith("DSS2"):
            return "dss"
        return "other"

    def load_image(choice, custom_id, ra, dec, fov, n):
        """Return (data, short name) for a dropdown choice or a custom HiPS ID."""
        if choice == CUSTOM_HIPS:
            hid = (custom_id or "").strip()
            if not hid:
                raise ValueError("choose a survey or type a HiPS ID")
            hdu, short = cached_hips(hid, ra, dec, fov, n), hid.split("/")[-1]
        elif choice in STSCI and fov * 60 <= STSCI_MAX_ARCMIN:
            hdu, short = cached_background(choice, ra, dec, fov, n), IMAGE_SURVEYS[choice][0]
        else:
            short, hid = IMAGE_SURVEYS[choice]
            hdu = cached_hips(hid, ra, dec, fov, n)
        data = np.nan_to_num(hdu.data.astype(float))
        if np.count_nonzero(data) == 0:
            raise ValueError("target is outside the survey footprint")
        return data, short

    def stretch(data, lo_pct=1.0, hi_pct=99.5):
        """sqrt stretch between two percentiles, clipped to 0..1 (safe for flat images)."""
        lo, hi = np.percentile(data, [lo_pct, hi_pct])
        if hi <= lo:
            return np.zeros_like(data)
        return np.sqrt(np.clip((data - lo) / (hi - lo), 0.0, 1.0))

    SCALINGS = [
        "automatic",
        "ROR (normalise + sqrt from 0.1σ)",
        "IOU (sqrt · sqrt · log)",
        "Optical (sqrt)",
        "Percentile sqrt (1–99.5 %)",
    ]

    def rgb_composite(chans, kinds, scaling, brightness=0.0, contrast=1.0):
        """Combine R, G, B arrays (None = empty channel) with rgbmaker's
        imagesopt=1 scalings, then apply brightness/contrast. Returns (n, n, 3) in 0..1."""
        from rgbmaker.imgplt import overlayc, overlayo

        present = [c for c in chans if c is not None]
        if not present:
            return np.zeros((1, 1, 3)), "none"
        filled = [c if c is not None else present[0] for c in chans]
        if scaling.startswith("rgbmaker (automatic"):
            k = [kd for kd, c in zip(kinds, chans) if c is not None]
            if "radio" in k:
                scaling = SCALINGS[1]
            elif all(kd == "dss" for kd in k):
                scaling = SCALINGS[3]
            else:
                scaling = SCALINGS[2]
        with np.errstate(all="ignore"):
            if scaling == SCALINGS[1]:
                img, _ = overlayc(filled[0], filled[1], filled[2], filled[0], 4, np.inf)
            elif scaling == SCALINGS[2]:
                img = overlayo(filled[0], filled[1], filled[2], kind="IOU")
            elif scaling == SCALINGS[3]:
                img = overlayo(filled[0], filled[1], filled[2], kind="Optical")
            else:
                img = np.dstack([stretch(c) for c in filled])
        img = np.nan_to_num(np.asarray(img, dtype=float))
        if img.max() > 1.0:
            img = img / 255.0
        for i, c in enumerate(chans):
            if c is None:
                img[..., i] = 0.0
        img = (img - 0.5) * contrast + 0.5 + brightness
        return np.clip(img, 0.0, 1.0), scaling.split(" (")[0]

    def contour_levels(data, floor, nsigma, nlev, use_floor):
        good = data[np.isfinite(data) & (data != 0)]
        _, _, rms = _clipped_stats(good, sigma=3, maxiters=5)
        low = floor if (floor is not None and use_floor) else nsigma * rms
        top = float(np.nanmax(data))
        if not np.isfinite(low) or low <= 0 or top <= low:
            return rms, None
        k = int(nlev)
        return rms, np.arange(low, top, (top - low) / k)[:k]  # linear, as rgbmaker

    def load_contours(labels, custom_id, ra, dec, fov, n, radius, nsigma, nlev, use_floor):
        chosen = [(lab, CONTOUR_SURVEYS[lab]) for lab in labels]
        if (custom_id or "").strip():
            hid = custom_id.strip()
            chosen.append((hid, dict(hips=hid, color="white", short=hid.split("/")[-1])))
        contours, rows, notes = [], [], []
        for lab, cfg in chosen:
            try:
                hdu = cached_hips(cfg["hips"], ra, dec, fov, n)
            except Exception as e:
                notes.append(f"{lab}: could not fetch ({e})")
                continue
            data = hdu.data.astype(float)
            if np.count_nonzero(data) == 0:
                notes.append(f"{lab}: target is outside the survey footprint")
                continue
            if cfg.get("smooth"):
                data = _gaussian(data, sigma=2 if radius < 0.12 else 1)
            rms, lv = contour_levels(data, cfg.get("floor"), nsigma, nlev, use_floor)
            unit = hdu.header.get("BUNIT", "") or "Jy/beam?"
            rows.append(dict(survey=lab, rms=f"{rms:.3g} {unit}",
                             levels=("none above threshold" if lv is None else
                                     ", ".join(f"{x:.3g}" for x in lv) + f" {unit}")))
            if lv is not None:
                contours.append((cfg, data, lv))
        return contours, rows, notes

    def draw_contour_set(ax, contours, extra_legend=None):
        handles, drawn = [], []
        for cfg, data, lv in contours:
            ax.contour(data, lv, colors=cfg["color"])
            handles.append(_mpatches.Patch(color=cfg["color"], label=cfg["short"].split("(")[0]))
            drawn.append(cfg["short"])
        if extra_legend:
            handles.append(_mpatches.Patch(color="white", label=extra_legend))
        if handles:
            ax.legend(handles=handles, labelcolor="linecolor", framealpha=0.0)
        return drawn

    return (
        CUSTOM_HIPS,
        IMAGE_SURVEYS,
        NO_CHANNEL,
        SCALINGS,
        draw_contour_set,
        load_contours,
        load_image,
        rgb_composite,
        stretch,
        survey_kind,
    )


@app.cell(hide_code=True)
def _(CONTOUR_SURVEYS, CUSTOM_HIPS, IMAGE_SURVEYS, NO_CHANNEL, SCALINGS, mo):
    # Custom mode controls. Plain controls (not a form), so switching the layout
    # shows or hides sections without resetting anything you have entered.
    _img = list(IMAGE_SURVEYS) + [CUSTOM_HIPS]
    _chan = [NO_CHANNEL] + _img
    _cont = list(CONTOUR_SURVEYS)

    def _hips_box():
        return mo.ui.text(placeholder="custom HiPS ID", full_width=False)

    def _extra_box():
        return mo.ui.text(placeholder="optional extra contour HiPS ID", full_width=True)

    def _brightness():
        return mo.ui.slider(start=-0.5, stop=0.5, step=0.05, value=0.0, show_value=True)

    def _contrast():
        return mo.ui.slider(start=0.2, stop=3.0, step=0.1, value=1.0, show_value=True)

    x_layout = mo.ui.radio(options=["Two panels", "Single panel"], value="Two panels", inline=True)
    x_single = mo.ui.radio(options=["RGB-C panel", "Composite contour panel"],
                           value="RGB-C panel", inline=True)
    x_target = mo.ui.dictionary(dict(
        name=mo.ui.text(value="Avi"),
        position=mo.ui.text(value="speca", full_width=True),
        radius=mo.ui.number(start=0.01, stop=2.0, step=0.01, value=0.12),
        px=mo.ui.dropdown(options=["240", "480", "720"], value="480"),
    ))
    x_rgb = mo.ui.dictionary(dict(
        r=mo.ui.dropdown(options=_chan, value=_cont[0]),
        r_custom=_hips_box(),
        g=mo.ui.dropdown(options=_chan, value="DSS2 Red"),
        g_custom=_hips_box(),
        b=mo.ui.dropdown(options=_chan, value=_cont[1]),
        b_custom=_hips_box(),
        scaling=mo.ui.dropdown(options=SCALINGS, value=SCALINGS[0]),
        brightness=_brightness(),
        contrast=_contrast(),
        contours=mo.ui.multiselect(options=_cont, value=[_cont[1]], full_width=True),
        contours_custom=_extra_box(),
    ))
    x_comp = mo.ui.dictionary(dict(
        bg=mo.ui.dropdown(options=_img, value="DSS2 Red"),
        bg_custom=_hips_box(),
        brightness=_brightness(),
        contrast=_contrast(),
        contours=mo.ui.multiselect(options=_cont, value=_cont[:3], full_width=True),
        contours_custom=_extra_box(),
    ))
    x_cset = mo.ui.dictionary(dict(
        nsigma=mo.ui.number(start=1, stop=50, step=0.5, value=5),
        nlev=mo.ui.number(start=1, stop=12, step=1, value=4),
        floors=mo.ui.checkbox(value=True, label="TGSS and NVSS: fixed lowest contour (15 and 1.5 mJy/beam) instead of N × rms"),
        axes=mo.ui.checkbox(value=False, label="Show RA/Dec axes"),
    ))
    x_run = mo.ui.run_button(label="Make custom image", kind="success")
    # the look of a bordered marimo form, for the Custom sections
    FORM_BOX = {
        "border": "1px solid var(--amber-6, #e2e8f0)",
        "border-radius": "8px",
        "padding": "0.75rem 1.25rem",
        "background": "var(--background, transparent)",
        "box-shadow": "0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.08)",
    }
    return FORM_BOX, x_comp, x_cset, x_layout, x_rgb, x_run, x_single, x_target


@app.cell
def _(x_layout, x_single):
    # which Custom panels are in use
    show_rgb_panel = x_layout.value == "Two panels" or x_single.value == "RGB-C panel"
    show_comp_panel = x_layout.value == "Two panels" or x_single.value == "Composite contour panel"
    return show_comp_panel, show_rgb_panel


@app.cell
def _(mo):
    # last Custom image, kept while settings change
    get_custom, set_custom = mo.state(None)
    return get_custom, set_custom


if __name__ == "__main__":
    app.run()